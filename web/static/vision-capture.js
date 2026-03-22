/**
 * 共享：MediaPipe → vision_frame → /ws/vision
 * 流程：先 getUserMedia（一次系统授权）→ play → 再加载模型 → 等 WS 连通 → 开环
 * 供 index.html（同页）与 vision.html 使用
 */
const MP_VER = "0.10.14";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`;
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

const LEFT_EAR_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EAR_IDX = [263, 386, 385, 362, 374, 380];
const MAR_IDX = [78, 81, 13, 311, 308, 402, 314, 405];
const FOREHEAD_IDX = [10, 151, 9, 175, 107, 336];

// 只传疲劳相关的 BlendShapes (控制 payload 大小)
const FATIGUE_BLENDSHAPES = new Set([
  "eyeBlinkLeft", "eyeBlinkRight",       // 眨眼 (替代 EAR)
  "eyeSquintLeft", "eyeSquintRight",     // 眯眼 (疲劳辅助)
  "eyeLookDownLeft", "eyeLookDownRight", // 眼球下看
  "jawOpen",                             // 张口 (替代 MAR)
  "mouthFunnel",                         // 嘴巴圆张 (区分哈欠与说话)
  "browDownLeft", "browDownRight",       // 皱眉 (压力/疲劳)
  "browInnerUp",                         // 眉毛上挑 (惊讶/挣扎清醒)
  "cheekPuff",                           // 鼓腮 (深呼吸/叹气)
]);

function dist2(lm, i, j) {
  const a = lm[i];
  const b = lm[j];
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function earFrom(lm, idx) {
  const [p1, p2, p3, p4, p5, p6] = idx;
  const num = dist2(lm, p2, p6) + dist2(lm, p3, p5);
  const den = 2 * dist2(lm, p1, p4) + 1e-8;
  return num / den;
}

function marFrom(lm) {
  const [m1, m2, m3, m4, m5, m6, m7, m8] = MAR_IDX;
  const num = dist2(lm, m2, m8) + dist2(lm, m3, m7) + dist2(lm, m4, m6);
  const den = 3 * dist2(lm, m1, m5) + 1e-8;
  return num / den;
}

function matrixToEulerDeg(data) {
  if (!data || data.length < 16) return { pitch: 0, yaw: 0, roll: 0 };
  const m = data;
  const r20 = m[8],
    r21 = m[9],
    r22 = m[10];
  const r10 = m[4],
    r00 = m[0];
  const pitch = (Math.asin(Math.max(-1, Math.min(1, -r20))) * 180) / Math.PI;
  const yaw = (Math.atan2(r21, r22) * 180) / Math.PI;
  const roll = (Math.atan2(r10, r00) * 180) / Math.PI;
  return { pitch, yaw, roll };
}

function sampleForeheadRgb(lm, w, h, ctx) {
  let sx = 0,
    sy = 0;
  for (const i of FOREHEAD_IDX) {
    sx += lm[i].x;
    sy += lm[i].y;
  }
  sx /= FOREHEAD_IDX.length;
  sy /= FOREHEAD_IDX.length;
  const rw = Math.max(16, Math.floor(w * 0.12));
  const rh = Math.max(12, Math.floor(h * 0.08));
  let cx = Math.floor(sx * w - rw / 2);
  let cy = Math.floor(sy * h - rh * 1.2);
  cx = Math.max(0, Math.min(w - rw, cx));
  cy = Math.max(0, Math.min(h - rh, cy));
  const data = ctx.getImageData(cx, cy, rw, rh).data;
  let r = 0,
    g = 0,
    b = 0,
    n = 0;
  for (let i = 0; i < data.length; i += 4) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
    n++;
  }
  if (!n) return { r: 0, g: 0, b: 0 };
  return { r: r / n / 255, g: g / n / 255, b: b / n / 255 };
}

async function loadLandmarker() {
  const { FaceLandmarker, FilesetResolver } = await import(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/+esm`
  );
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL,
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
}

function openVisionWebSocket(timeoutMs = 12000) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws/vision`;
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    const to = setTimeout(() => {
      try {
        socket.close();
      } catch (_) {}
      reject(new Error(`WS 连接超时（${timeoutMs / 1000}s）`));
    }, timeoutMs);
    socket.addEventListener("open", () => {
      clearTimeout(to);
      resolve(socket);
    });
    socket.addEventListener("error", () => {
      clearTimeout(to);
      reject(new Error("WebSocket 错误（请确认本页与网关同源、服务已启动）"));
    });
  });
}

/**
 * @param {object} opts
 * @param {HTMLVideoElement} opts.video
 * @param {HTMLCanvasElement} opts.canvas
 * @param {(s: string) => void} [opts.onStatus]
 * @param {(s: string) => void} [opts.onWsState]
 * @param {number} [opts.sendIntervalMs]
 */
export function createVisionCapture(opts) {
  const video = opts.video;
  const canvas = opts.canvas;
  const onStatus = opts.onStatus || (() => {});
  const onWsState = opts.onWsState || (() => {});
  const SEND_MS = opts.sendIntervalMs ?? 50;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  let ws = null;
  let landmarker = null;
  let seq = 0;
  let rafId = null;
  let lastSend = 0;
  let running = false;
  let t0 = performance.now() / 1000;
  let lastVideoTime = -1;

  function sendFrame(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function loop() {
    if (!running) return;
    rafId = requestAnimationFrame(loop);
    if (video.readyState < 2 || !landmarker) return;

    const nowMs = performance.now();
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;

    const result = landmarker.detectForVideo(video, nowMs);
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w && h && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
    }
    if (w && h) {
      ctx.drawImage(video, 0, 0, w, h);
    }

    const t = t0 + nowMs / 1000;
    seq += 1;
    const tab = {
      visible: document.visibilityState === "visible",
      hostname: location.hostname || "vision",
    };
    const capture = { width: w || 0, height: h || 0, fps_hint: 30 };

    const faces = result.faceLandmarks || [];
    if (!faces.length) {
      if (nowMs - lastSend >= SEND_MS) {
        lastSend = nowMs;
        sendFrame({
          type: "vision_frame",
          schema: 1,
          t,
          seq,
          tab,
          capture,
          face: { present: false, confidence: 0 },
        });
      }
      return;
    }

    const lm = faces[0];
    const faceConf =
      result.faceLandmarks.length > 0
        ? Math.min(
            1,
            Math.max(
              0,
              1 -
                result.faceLandmarks[0].reduce(
                  (s, p) => s + (p.visibility !== undefined ? 1 - p.visibility : 0),
                  0
                ) /
                  result.faceLandmarks[0].length
            )
          )
        : 0.9;

    const earLeft = earFrom(lm, LEFT_EAR_IDX);
    const earRight = earFrom(lm, RIGHT_EAR_IDX);
    const mar = marFrom(lm);
    let headPoseDeg = { pitch: 0, yaw: 0, roll: 0 };
    const mats = result.facialTransformationMatrixes;
    if (mats && mats.length && mats[0].data) {
      headPoseDeg = matrixToEulerDeg(Array.from(mats[0].data));
    }

    // BlendShapes: 只提取疲劳相关的系数 (精简 payload)
    let blendshapes;
    const bsRaw = result.faceBlendshapes;
    if (bsRaw && bsRaw.length && bsRaw[0].categories) {
      blendshapes = {};
      for (const cat of bsRaw[0].categories) {
        if (FATIGUE_BLENDSHAPES.has(cat.categoryName)) {
          blendshapes[cat.categoryName] = Math.round(cat.score * 1000) / 1000;
        }
      }
    }

    let skin;
    if (w && h) {
      try {
        skin = { roiRgb: sampleForeheadRgb(lm, w, h, ctx) };
      } catch (_) {}
    }

    if (nowMs - lastSend < SEND_MS) return;
    lastSend = nowMs;

    sendFrame({
      type: "vision_frame",
      schema: 2,
      t,
      seq,
      tab,
      capture,
      face: { present: true, confidence: faceConf },
      geometry: {
        earLeft,
        earRight,
        ear: Math.min(earLeft, earRight),
        mar,
        headPoseDeg,
      },
      ...(blendshapes ? { blendshapes } : {}),
      ...(skin ? { skin } : {}),
    });
  }

  async function start() {
    if (running) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("当前环境不支持 getUserMedia（需 HTTPS 或 localhost）");
    }

    video.setAttribute("playsinline", "true");
    video.setAttribute("muted", "true");
    video.muted = true;

    onStatus("请在系统对话框中允许摄像头…");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (e) {
      const name = e && e.name;
      const msg =
        name === "NotAllowedError" || name === "PermissionDeniedError"
          ? "摄像头权限被拒绝，请在浏览器设置中允许本站使用摄像头。"
          : name === "NotFoundError"
            ? "未找到摄像头设备。"
            : String(e.message || e);
      throw new Error(msg);
    }

    video.srcObject = stream;
    try {
      await video.play();
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
      throw new Error("视频无法播放: " + (e.message || e) + "（请保持本页在前台重试）");
    }

    onStatus("摄像头已就绪，正在加载识别模型（首次需数秒）…");
    try {
      landmarker = await loadLandmarker();
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
      throw new Error("模型加载失败: " + (e.message || e));
    }

    onWsState("连接中…");
    try {
      ws = await openVisionWebSocket();
    } catch (e) {
      landmarker = null;
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
      throw e;
    }
    ws.addEventListener("close", () => onWsState("已断开"));
    ws.addEventListener("error", () => onWsState("异常"));
    onWsState("已连接");

    t0 = Date.now() / 1000 - performance.now() / 1000;
    lastVideoTime = -1;
    running = true;
    loop();
    onStatus("推流中 · 可回到本页继续使用（勿关标签页）");
  }

  function stop() {
    running = false;
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    landmarker = null;
    if (video.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    if (ws) {
      try {
        ws.close();
      } catch (_) {}
      ws = null;
    }
    onWsState("未连接");
    onStatus("已停止摄像头推流。");
  }

  return {
    start,
    stop,
    get isRunning() {
      return running;
    },
  };
}
