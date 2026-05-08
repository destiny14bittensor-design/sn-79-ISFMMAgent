"""Order stream constants and parquet schema.

Shared across all data sources (data collection and later offline dataprocessing)
and consumers (tokenizer, dataloader, training).
"""

import pyarrow as pa

# Order types
BID = 0
ASK = 1
CANCEL = 2

# LOB depth: number of price levels tracked per side
LOB_DEPTH = 10


def order_stream_schema() -> pa.Schema:
    """Parquet schema for order stream files.

    All data sources (Kraken, simulation, live collection) produce this schema.
    The dataloader and tokenizer consume it.
    """
    fields = [
        pa.field("timestamp", pa.timestamp("ns")),
        pa.field("order_type", pa.int8()),  # 0=Bid, 1=Ask, 2=Cancel
        pa.field("rel_price", pa.int64()),  # relative to mid, integer ticks
        pa.field("volume_int", pa.int32()),  # integer part of qty
        pa.field("volume_dec", pa.float32()),  # fractional part [0.0, 1.0)
        pa.field("interval_ns", pa.int64()),  # ns since previous order
        pa.field("mid_price", pa.int64()),  # absolute mid in integer ticks
        pa.field("time_of_day_s", pa.int32()),  # seconds since midnight UTC
        pa.field("mid_price_delta", pa.int64()),  # mid change from session open (ticks)
    ]
    for side in ("ask", "bid"):
        for i in range(1, LOB_DEPTH + 1):
            fields.append(pa.field(f"lob_{side}_vol_{i}", pa.float64()))
    return pa.schema(fields)
