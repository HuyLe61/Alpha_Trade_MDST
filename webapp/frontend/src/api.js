const BASE = "";

export async function fetchStocks() {
  const res = await fetch(`${BASE}/api/stocks`);
  return res.json();
}

export async function fetchAlgorithms() {
  const res = await fetch(`${BASE}/api/algorithms`);
  return res.json();
}

export async function startTraining(payload) {
  const res = await fetch(`${BASE}/api/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function getJobStatus(jobId) {
  const res = await fetch(`${BASE}/api/jobs/${jobId}`);
  return res.json();
}

export function connectTrainingWs(jobId, onMessage) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/train/${jobId}`);
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch {}
  };
  return ws;
}
