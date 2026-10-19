async function errorMessage(response: Response): Promise<string> {
  const body = await response.json().catch(() => null);
  if (!body) return response.statusText;
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail.map((item: { msg?: string }) => item.msg || "invalid").join("; ");
  }
  return response.statusText;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json() as Promise<T>;
}

export function sessionSocket(sessionId: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${protocol}://${window.location.host}/api/v1/ws/sessions/${sessionId}`);
}
