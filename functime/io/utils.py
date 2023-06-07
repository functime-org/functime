from functime.io.client import FunctimeH2Client


def get_stubs():
    with FunctimeH2Client() as client:
        response = client.get(
            "/list_models",
        )
    return response.json()
