"""Smoke tests for Q-NeuroDetect Parkinson API – Phase 1."""
import io
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "models_loaded" in body


@pytest.mark.anyio
async def test_predict_voice_mock(client):
    """Minimal WAV header bytes – preprocessing will fall back to mock."""
    fake_wav = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 50)
    r = await client.post(
        "/predict/voice",
        files={"file": ("test.wav", fake_wav, "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "voice"
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("Parkinson", "Healthy")


@pytest.mark.anyio
async def test_predict_mri_mock(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(128, 64, 200)).save(buf, format="PNG")
    buf.seek(0)
    r = await client.post(
        "/predict/mri",
        files={"file": ("scan.png", buf, "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["modality"] == "mri"


@pytest.mark.anyio
async def test_predict_drawing_mock(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (64, 64), color=180).save(buf, format="PNG")
    buf.seek(0)
    r = await client.post(
        "/predict/drawing",
        files={"file": ("spiral.png", buf, "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["modality"] == "drawing"


@pytest.mark.anyio
async def test_predict_all_no_files(client):
    """Fusion endpoint must return a valid response even with no files."""
    r = await client.post("/predict/all")
    assert r.status_code == 200
    body = r.json()
    assert "fusion" in body
    assert "modalities" in body
    assert "model_comparison" in body
    assert "explanation" in body


@pytest.mark.anyio
async def test_invalid_voice_extension(client):
    r = await client.post(
        "/predict/voice",
        files={"file": ("audio.txt", io.BytesIO(b"nope"), "text/plain")},
    )
    assert r.status_code == 422
