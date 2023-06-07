import pyarrow as pa


def serialize_bytes(table: pa.Table) -> bytes:
    with pa.BufferOutputStream() as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        bytes_data = sink.getvalue().to_pybytes()
    return bytes_data


def deserialize_bytes(bytes_data: bytes) -> pa.Table:
    with pa.BufferReader(bytes_data) as source:
        with pa.ipc.open_stream(source) as reader:
            table = reader.read_all()
    return table
