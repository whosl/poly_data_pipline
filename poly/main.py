"""CLI entry point for the Polymarket data collection system."""

from __future__ import annotations

import asyncio
import signal

import click
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# collect command
# ---------------------------------------------------------------------------

async def _run_collectors(tags_str: str, symbols_str: str, raw_only: bool) -> None:
    from poly.config import get_config
    from poly.storage.raw import RawWriter
    from poly.engine.orderbook import OrderBookEngine
    from poly.collector.user_ws import PolymarketUserWS
    from poly.collector.binance_ws import BinanceWS
    from poly.collector.updown_ws import UpDownCollector

    config = get_config()
    symbols = [s.strip() for s in symbols_str.split(",")]

    # Create raw writers
    poly_raw = RawWriter(config.data_dir, "polymarket", "market",
                         config.raw_flush_interval)
    user_raw = RawWriter(config.data_dir, "polymarket", "user",
                         config.raw_flush_interval)
    binance_raw = RawWriter(config.data_dir, "binance", "spot",
                            config.raw_flush_interval)

    # Create normalized writers only if not raw-only
    if not raw_only:
        from poly.storage.normalized import (
            ParquetWriter, L2_BOOK_SCHEMA, TRADE_SCHEMA,
            BEST_BID_ASK_SCHEMA, ORDER_SCHEMA, USER_TRADE_SCHEMA,
        )
        poly_book_w = ParquetWriter(L2_BOOK_SCHEMA, config.data_dir,
                                    "poly_l2_book", config.norm_buffer_size)
        poly_trade_w = ParquetWriter(TRADE_SCHEMA, config.data_dir,
                                     "poly_trades", config.norm_buffer_size)
        poly_bba_w = ParquetWriter(BEST_BID_ASK_SCHEMA, config.data_dir,
                                   "poly_best_bid_ask", config.norm_buffer_size)
        order_w = ParquetWriter(ORDER_SCHEMA, config.data_dir,
                                "poly_orders", config.norm_buffer_size)
        user_trade_w = ParquetWriter(USER_TRADE_SCHEMA, config.data_dir,
                                     "poly_user_trades", config.norm_buffer_size)
        binance_bba_w = ParquetWriter(BEST_BID_ASK_SCHEMA, config.data_dir,
                                      "binance_best_bid_ask", config.norm_buffer_size)
        binance_trade_w = ParquetWriter(TRADE_SCHEMA, config.data_dir,
                                        "binance_trades", config.norm_buffer_size)
        binance_book_w = ParquetWriter(L2_BOOK_SCHEMA, config.data_dir,
                                       "binance_l2_book", config.norm_buffer_size)
        engine = OrderBookEngine()
        parquet_writers = [poly_book_w, poly_trade_w, poly_bba_w, order_w,
                           user_trade_w, binance_bba_w, binance_trade_w, binance_book_w]
    else:
        poly_book_w = poly_trade_w = poly_bba_w = None
        order_w = user_trade_w = None
        binance_bba_w = binance_trade_w = binance_book_w = None
        engine = None
        parquet_writers = []

    # UpDown collector: configured 5m/15m markets with auto-rotation
    updown_collector = UpDownCollector(
        config, poly_raw, poly_book_w, poly_trade_w, poly_bba_w, engine,
        raw_only=raw_only,
    )
    user_ws = PolymarketUserWS(config, user_raw, order_w, user_trade_w)
    binance_ws = BinanceWS(
        config, binance_raw, binance_bba_w, binance_trade_w, binance_book_w,
        raw_only=raw_only,
    )

    # Graceful shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    # Launch tasks
    tasks = [
        asyncio.create_task(updown_collector.run(), name="updown_ws"),
        asyncio.create_task(user_ws.run([]), name="user_ws"),
        asyncio.create_task(binance_ws.run(symbols), name="binance_ws"),
        asyncio.create_task(poly_raw.flush_loop(), name="poly_flush"),
        asyncio.create_task(user_raw.flush_loop(), name="user_flush"),
        asyncio.create_task(binance_raw.flush_loop(), name="binance_flush"),
    ]

    logger.info("collectors_started")
    await shutdown_event.wait()

    # Shutdown
    logger.info("shutting_down")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Flush all writers
    await poly_raw.close()
    await user_raw.close()
    await binance_raw.close()
    for w in parquet_writers:
        w.close()
    logger.info("shutdown_complete")


# ---------------------------------------------------------------------------
# normalize command
# ---------------------------------------------------------------------------

async def _run_normalize(date: str, source: str) -> None:
    from poly.config import get_config
    from poly.normalize.poly_norm import PolyNormalizer
    from poly.normalize.binance_norm import BinanceNormalizer

    config = get_config()
    if source in ("all", "polymarket"):
        PolyNormalizer().run(config.data_dir, date)
    if source in ("all", "binance"):
        BinanceNormalizer().run(config.data_dir, date)


# ---------------------------------------------------------------------------
# metadata command
# ---------------------------------------------------------------------------

async def _run_metadata(dates: tuple[str, ...]) -> None:
    from poly.config import get_config
    from poly.metadata.polymarket import MetadataFetchConfig, fetch_metadata, write_metadata_by_date
    from poly.training.io import discover_dates

    config = get_config()
    target_dates = discover_dates(config.data_dir, dates or None)
    if not target_dates:
        raise click.ClickException("No dates found under normalized/research. Pass at least one YYYYMMDD date.")
    fetch_config = MetadataFetchConfig(
        data_dir=config.data_dir,
        gamma_url=config.gamma_url,
        clob_url=config.clob_url,
        dates=tuple(target_dates),
    )
    frame = await fetch_metadata(fetch_config)
    paths = write_metadata_by_date(frame, config.data_dir, target_dates)
    click.echo(f"rows={frame.height}")
    for path in paths:
        click.echo(f"metadata={path}")


# ---------------------------------------------------------------------------
# labels command
# ---------------------------------------------------------------------------

async def _run_labels(date: str) -> None:
    from poly.config import get_config
    from poly.normalize.labels import LabelGenerator

    config = get_config()
    LabelGenerator().run(config.data_dir, date)


# ---------------------------------------------------------------------------
# replay command
# ---------------------------------------------------------------------------

async def _run_replay(source: str, date: str, speed: float) -> None:
    from poly.config import get_config
    from poly.replay.reader import ReplayPlayer

    config = get_config()
    player = ReplayPlayer(config.data_dir)
    await player.replay(source, date, speed)


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

def _show_status() -> None:
    from poly.config import get_config

    config = get_config()
    data_dir = config.data_dir
    print(f"Data directory: {data_dir}")

    if not data_dir.exists():
        print("  (not yet created)")
        return

    for subdir in ["raw_feed", "normalized", "research"]:
        d = data_dir / subdir
        if d.exists():
            dates = sorted(p.name for p in d.iterdir() if p.is_dir())
            print(f"  {subdir}: {len(dates)} date(s) — {dates[:5]}")
        else:
            print(f"  {subdir}: (empty)")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Polymarket low-latency quantitative data collection system."""


@cli.command()
@click.option("--tags", default="bitcoin,crypto", help="Comma-separated market tags (unused, kept for compat)")
@click.option("--symbols", default="btcusdt", help="Comma-separated Binance symbols")
@click.option("--raw-only", is_flag=True, help="Only save raw JSONL, skip normalized Parquet (lower CPU)")
def collect(tags, symbols, raw_only):
    """Start all data collectors (UpDown markets + Binance)."""
    asyncio.run(_run_collectors(tags, symbols, raw_only))


@cli.command()
@click.argument("date")
@click.option("--source", default="all", help="polymarket, binance, or all")
def normalize(date, source):
    """Run normalization pipeline for a specific date."""
    asyncio.run(_run_normalize(date, source))


@cli.command()
@click.argument("dates", nargs=-1)
def metadata(dates):
    """Fetch Polymarket Up/Down market metadata parquet."""
    asyncio.run(_run_metadata(tuple(dates)))


@cli.command()
@click.argument("date")
def labels(date):
    """Generate research labels for a specific date."""
    asyncio.run(_run_labels(date))


@cli.command()
@click.argument("source")
@click.argument("date")
@click.option("--speed", default=1.0, help="Replay speed (0=max)")
def replay(source, date, speed):
    """Replay raw data at original or adjusted speed."""
    asyncio.run(_run_replay(source, date, speed))


@cli.command()
@click.argument("date")
@click.option("--layer", default="all", help="raw, normalized, research, or all")
@click.confirmation_option(prompt="Are you sure you want to delete this data?")
def purge(date, layer):
    """Delete collected data for a specific date."""
    _purge_data(date, layer)


def _purge_data(date: str, layer: str) -> None:
    import shutil
    from poly.config import get_config

    config = get_config()
    data_dir = config.data_dir
    deleted = []

    layers = {
        "raw": data_dir / "raw_feed" / date,
        "normalized": data_dir / "normalized" / date,
        "research": data_dir / "research" / date,
    }

    if layer == "all":
        targets = layers
    elif layer in layers:
        targets = {layer: layers[layer]}
    else:
        print(f"Unknown layer: {layer}. Use raw, normalized, research, or all.")
        return

    for name, path in targets.items():
        if path.exists():
            size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
            shutil.rmtree(path)
            deleted.append(f"{name}: {size_mb:.1f} MB")
        else:
            print(f"  {name}: not found, skipping")

    if deleted:
        print(f"Deleted {date}:")
        for d in deleted:
            print(f"  {d}")
    else:
        print(f"No data found for {date}")


@cli.command()
def status():
    """Show data directory status."""
    _show_status()


if __name__ == "__main__":
    cli()
