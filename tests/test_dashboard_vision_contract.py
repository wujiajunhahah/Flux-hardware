from fastapi.testclient import TestClient

import web.app as web_app


def test_dashboard_index_exposes_vision_metrics_panel():
    with TestClient(web_app.app) as client:
        resp = client.get("/")

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store, max-age=0"
    html = resp.text
    assert '/static/dashboard.css?v=' in html
    assert '/static/dashboard.js?v=' in html
    for marker in (
        'id="visionPanel"',
        'id="visionMetrics"',
        'id="visionBlinkRate"',
        'id="visionPerclos"',
        'id="visionYawns"',
        'id="visionAlertness"',
        'id="visionFaceState"',
        'id="visionPoseState"',
    ):
        assert marker in html


def test_dashboard_css_preserves_full_camera_frame():
    with TestClient(web_app.app) as client:
        resp = client.get("/static/dashboard.css")

    assert resp.status_code == 200
    css = resp.text
    assert "object-fit: contain;" in css
    assert "aspect-ratio: 4 / 3;" in css


def test_dashboard_script_renders_vision_snapshot_fields():
    with TestClient(web_app.app) as client:
        resp = client.get("/static/dashboard.js")

    assert resp.status_code == 200
    script = resp.text
    assert "function renderVisionPanel" in script
    assert "async function refreshLiveState" in script
    assert '$("visionBlinkRate")' in script
    assert '$("visionPerclos")' in script
    assert '$("visionYawns")' in script
    assert '$("visionAlertness")' in script
    assert "renderVisionPanel(payload?.vision)" in script
    assert "await refreshLiveState();" in script
    assert "window.setInterval(refreshLiveState, 2000);" in script
