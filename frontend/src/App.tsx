import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, sessionSocket } from "./api";
import { RocketView } from "./simulator/RocketView";
import type { Analytics, Checkpoint, Experiment, Frame, Hello, Trajectory, TrajectorySummary } from "./types";

type Page = "overview" | "simulator" | "training" | "analytics" | "replay" | "compare";

const NAV: { id: Page; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "simulator", label: "Live Simulator" },
  { id: "training", label: "Training" },
  { id: "analytics", label: "Analytics" },
  { id: "replay", label: "Flight Replay" },
  { id: "compare", label: "Comparison" },
];

export function App() {
  const [page, setPage] = useState<Page>("overview");
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => {
    api("/api/v1/health")
      .then(() => setOnline(true))
      .catch(() => setOnline(false));
  }, [page]);
  return (
    <div className="app">
      <nav className="nav">
        <div className="brand">
          <strong>LUNARMIND</strong>
          <span>Landing research console</span>
        </div>
        {NAV.map((item) => (
          <button key={item.id} className={page === item.id ? "active" : ""} onClick={() => setPage(item.id)}>
            {item.label}
          </button>
        ))}
      </nav>
      <main className="main">
        {online === false && <p className="error">The API is not reachable. Start both servers with python scripts/dev.py.</p>}
        {page === "overview" && <Overview onNavigate={setPage} />}
        {page === "simulator" && <Simulator />}
        {page === "training" && <Training />}
        {page === "analytics" && <AnalyticsPage />}
        {page === "replay" && <Replay />}
        {page === "compare" && <Compare />}
      </main>
    </div>
  );
}

function Overview({ onNavigate }: { onNavigate: (page: Page) => void }) {
  const [meta, setMeta] = useState<Record<string, unknown> | null>(null);
  const [overview, setOverview] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([api<Record<string, unknown>>("/api/v1/meta"), api<Record<string, unknown>>("/api/v1/overview")])
      .then(([metaBody, overviewBody]) => {
        setMeta(metaBody);
        setOverview(overviewBody);
      })
      .catch((exc: Error) => setError(exc.message));
  }, []);
  const reward = (meta?.reward || {}) as Record<string, unknown>;
  const active = overview?.active_experiment as Experiment | null | undefined;
  return (
    <>
      <header className="top">
        <div>
          <h1>Overview</h1>
          <p>Fly the rocket, watch an untrained policy fail, then train a PPO agent on the same physics.</p>
        </div>
        <span className={`pill ${active?.status || ""}`}>{active ? `${active.name}: ${active.status}` : "no experiment yet"}</span>
      </header>
      {error && <p className="error">{error}</p>}
      <div className="grid two">
        <section className="panel checklist">
          <h2>First flight</h2>
          <button onClick={() => onNavigate("simulator")}>1. Fly it yourself with the arrow keys and space</button>
          <button onClick={() => onNavigate("simulator")}>2. Watch a random agent, then the safe and crash physics demonstrations</button>
          <button onClick={() => onNavigate("training")}>3. Start a PPO run and keep the step-0 checkpoint</button>
          <button onClick={() => onNavigate("analytics")}>4. Read the reward curve from stored training episodes</button>
          <button onClick={() => onNavigate("replay")}>5. Replay a recorded flight and its reward breakdown</button>
          <button onClick={() => onNavigate("compare")}>6. Compare human flights, random actions and checkpoints on the same starts</button>
        </section>
        <section className="panel">
          <h2>Scoring rules</h2>
          <p className="mono">distance: {String(reward.distance || "")}</p>
          <p className="mono">speed: {String(reward.speed || "")}</p>
          <p className="mono">upright: {String(reward.upright || "")}</p>
          <p className="mono">landing: {String(reward.landing_bonus ?? "")} · crash: {String(reward.crash_penalty ?? "")}</p>
          <p className="note">Fuel and time are not scored. Truncation is neither a landing nor a crash. The language-model controller, when configured, is a prompted baseline rather than a trained policy.</p>
        </section>
      </div>
    </>
  );
}

function Simulator() {
  const [mode, setMode] = useState("human");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [checkpointId, setCheckpointId] = useState("");
  const [hello, setHello] = useState<Hello | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);
  const socketRef = useState<{ current: WebSocket | null }>(() => ({ current: null }))[0];
  const keysRef = useState<{ order: string[] }>(() => ({ order: [] }))[0];

  useEffect(() => {
    api<Checkpoint[]>("/api/v1/checkpoints").then(setCheckpoints).catch(() => setCheckpoints([]));
  }, []);

  useEffect(() => {
    const sendKeys = () => {
      const socket = socketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({ type: "keys", pressed: keysRef.order }));
    };
    const down = (event: KeyboardEvent) => {
      if (!["ArrowLeft", "ArrowRight", "Space"].includes(event.key)) return;
      event.preventDefault();
      if (event.repeat) return;
      keysRef.order = [...keysRef.order.filter((key) => key !== event.key), event.key];
      sendKeys();
    };
    const up = (event: KeyboardEvent) => {
      if (!keysRef.order.includes(event.key)) return;
      keysRef.order = keysRef.order.filter((key) => key !== event.key);
      sendKeys();
    };
    const blur = () => {
      keysRef.order = [];
      sendKeys();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
      const socket = socketRef.current;
      socketRef.current = null;
      socket?.close();
    };
  }, [keysRef, socketRef]);

  function connect() {
    setError("");
    const previous = socketRef.current;
    socketRef.current = null;
    if (previous) {
      previous.onclose = null;
      previous.close();
    }
    const demo = mode.startsWith("demo-") ? mode.slice(5) : null;
    const controller = demo ? "demo" : mode;
    api<{ session_id: string }>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({
        controller,
        demo,
        checkpoint_id: mode === "ppo" ? checkpointId : null,
        seed: 0,
        speed: 1,
      }),
    })
      .then((session) => {
        const socket = sessionSocket(session.session_id);
        socketRef.current = socket;
        socket.onmessage = (event) => {
          const message = JSON.parse(event.data);
          if (message.type === "hello") setHello(message);
          if (message.type === "frame") setFrame(message);
        };
        socket.onopen = () => {
          if (socketRef.current !== socket) return;
          setConnected(true);
          if (keysRef.order.length) {
            socket.send(JSON.stringify({ type: "keys", pressed: keysRef.order }));
          }
        };
        socket.onclose = () => {
          if (socketRef.current === socket) setConnected(false);
        };
        socket.onerror = () => {
          if (socketRef.current === socket) setError("The flight socket closed.");
        };
      })
      .catch((exc: Error) => setError(exc.message));
  }

  function send(message: object) {
    socketRef.current?.send(JSON.stringify(message));
  }

  const telemetry = frame?.telemetry;
  return (
    <>
      <header className="top">
        <div>
          <h1>Live simulator</h1>
          <p>Click the viewport, then hold Left, Right and Space. One thruster fires at a time; the newest key wins.</p>
        </div>
        <span className={`pill ${connected ? "running" : ""}`}>{connected ? "live" : "idle"}</span>
      </header>
      <div className="row" style={{ marginBottom: 12 }}>
        <label>
          Controller
          <select value={mode} onChange={(event) => setMode(event.target.value)}>
            <option value="human">Human</option>
            <option value="random">Random</option>
            <option value="ppo">PPO checkpoint</option>
            <option value="llm">LLM baseline</option>
            <option value="demo-safe">Physics demo: gentle drop</option>
            <option value="demo-crash">Physics demo: fast impact</option>
            <option value="demo-tilt">Physics demo: tilted impact</option>
          </select>
        </label>
        {mode === "ppo" && (
          <label>
            Checkpoint
            <select value={checkpointId} onChange={(event) => setCheckpointId(event.target.value)}>
              <option value="">Select a checkpoint</option>
              {checkpoints.map((item) => (
                <option key={item.id} value={item.id}>
                  {(item.experiment_name || item.experiment_id) + " · " + item.timesteps + " steps"}
                </option>
              ))}
            </select>
          </label>
        )}
        <button className="btn primary" onClick={connect}>
          {connected ? "Restart flight" : "Start flight"}
        </button>
        <button className="btn" disabled={!connected} onClick={() => send({ type: "pause", paused: !frame?.paused })}>
          {frame?.paused ? "Resume" : "Pause"}
        </button>
        <button className="btn" disabled={!connected} onClick={() => send({ type: "reset" })}>
          Reset
        </button>
        {[0.5, 1, 2, 4].map((speed) => (
          <button key={speed} className="btn" disabled={!connected} onClick={() => send({ type: "speed", value: speed })}>
            {speed}×
          </button>
        ))}
      </div>
      {error && <p className="error">{error}</p>}
      {mode === "llm" && <p className="note">The LLM controller needs OPENAI_API_KEY. Without it, the rocket coasts and the trace records llm_unconfigured. It does not learn.</p>}
      <div className="grid two">
        <RocketView
          terrain={hello?.terrain || []}
          pad={hello?.pad || { x: 0, half_width: 4 }}
          geometry={frame?.geometry || null}
          trail={frame?.trail || []}
        />
        <section className="panel">
          <h2>{frame?.model_label || hello?.model_label || "No flight"}</h2>
          <div className="horizon" aria-hidden>
            <i style={{ transform: `rotate(${-(telemetry?.signed_angle_rad || 0)}rad)` }} />
          </div>
          <div className="telemetry" style={{ marginTop: 12 }}>
            <Stat label="Horizontal position" value={fmt(telemetry?.x_m, "m")} />
            <Stat label="Altitude" value={fmt(telemetry?.altitude_m, "m")} />
            <Stat label="Horizontal velocity" value={fmt(telemetry?.vx_mps, "m/s")} />
            <Stat label="Vertical velocity" value={fmt(telemetry?.vy_mps, "m/s")} />
            <Stat label="Total speed" value={fmt(telemetry?.speed_mps, "m/s")} />
            <Stat label="Tilt" value={fmt(telemetry?.tilt_deg, "deg")} />
            <Stat label="Angular velocity" value={fmt(telemetry?.omega_radps, "rad/s")} />
            <Stat label="Action" value={telemetry?.action_name || "—"} />
            <Stat label="Step reward" value={fmt(telemetry?.step_reward)} />
            <Stat label="Episode reward" value={fmt(telemetry?.episode_reward)} />
            <Stat label="Distance to pad" value={fmt(telemetry?.distance_m, "m")} />
            <Stat label="Landing status" value={telemetry?.landing_status || "—"} />
            <Stat label="Controller" value={telemetry?.controller || "—"} />
          </div>
          <RewardList parts={frame?.reward_parts} />
        </section>
      </div>
    </>
  );
}

function Training() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState("");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [error, setError] = useState("");
  const [name, setName] = useState("baseline");
  const [timesteps, setTimesteps] = useState(50000);
  const [seed, setSeed] = useState(0);
  const [gravity, setGravity] = useState(1.62);
  const [rewardMode, setRewardMode] = useState("delta");

  function refresh(prefer?: string) {
    api<Experiment[]>("/api/v1/experiments")
      .then((rows) => {
        setExperiments(rows);
        setSelected((current) => prefer || current || rows[0]?.id || "");
      })
      .catch((exc: Error) => setError(exc.message));
  }
  useEffect(refresh, []);
  useEffect(() => {
    if (!selected) return;
    api<Checkpoint[]>(`/api/v1/experiments/${selected}/checkpoints`).then(setCheckpoints).catch(() => setCheckpoints([]));
  }, [selected, experiments]);

  const experiment = experiments.find((item) => item.id === selected);
  useEffect(() => {
    if (!experiment || !["running", "pause_requested", "stop_requested"].includes(experiment.status)) return;
    const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, [experiment?.status, experiment?.id]);

  function create() {
    setError("");
    api<Experiment>("/api/v1/experiments", {
      method: "POST",
      body: JSON.stringify({ name, timesteps, seed, gravity, reward_mode: rewardMode, auto_eval_episodes: 3 }),
    })
      .then((created) => refresh(created.id))
      .catch((exc: Error) => setError(exc.message));
  }
  function act(path: string) {
    setError("");
    api(`/api/v1/experiments/${selected}/${path}`, { method: "POST" })
      .then(() => refresh())
      .catch((exc: Error) => setError(exc.message));
  }

  const status = experiment?.status || "none";
  return (
    <>
      <header className="top">
        <div>
          <h1>Training</h1>
          <p>PPO runs in a separate process. Buttons follow the job status returned by the API.</p>
        </div>
        <span className={`pill ${status}`}>{status}</span>
      </header>
      {error && <p className="error">{error}</p>}
      <div className="grid two">
        <section className="panel">
          <h2>New experiment</h2>
          <div className="row">
            <label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
            <label>Steps<input type="number" value={timesteps} onChange={(event) => setTimesteps(Number(event.target.value))} /></label>
            <label>Seed<input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>
            <label>Gravity<input type="number" step="0.01" value={gravity} onChange={(event) => setGravity(Number(event.target.value))} /></label>
            <label>
              Reward
              <select value={rewardMode} onChange={(event) => setRewardMode(event.target.value)}>
                <option value="delta">Delta baseline</option>
                <option value="potential">Potential-based, optional</option>
              </select>
            </label>
            <button className="btn primary" onClick={create}>Create</button>
          </div>
          <p className="note">Milestones inside the step budget are saved, including the untrained step-0 network. Terminal rewards stay at +100 and −1000.</p>
        </section>
        <section className="panel">
          <h2>Job</h2>
          <label>
            Experiment
            <select value={selected} onChange={(event) => setSelected(event.target.value)}>
              {experiments.map((item) => (
                <option key={item.id} value={item.id}>{item.name} · {item.status}</option>
              ))}
            </select>
          </label>
          <p className="mono">{experiment ? `${experiment.timesteps_completed} / ${experiment.timesteps_target} steps` : "Create an experiment to enable training."}</p>
          <div className="row">
            <button className="btn primary" disabled={!experiment || !["created", "failed", "stopped"].includes(status)} onClick={() => act("start")}>Start</button>
            <button className="btn" disabled={status !== "running"} onClick={() => act("pause")}>Pause</button>
            <button className="btn" disabled={!experiment || !["paused", "stopped", "failed"].includes(status)} onClick={() => act("resume")}>Resume</button>
            <button className="btn danger" disabled={!["running", "paused", "pause_requested"].includes(status)} onClick={() => act("stop")}>Stop</button>
          </div>
          {experiment?.error && <p className="error">{experiment.error}</p>}
        </section>
      </div>
      <section className="panel" style={{ marginTop: 14 }}>
        <h2>Checkpoints</h2>
        {checkpoints.length === 0 && <p>No checkpoints yet. A step-0 file appears when training starts.</p>}
        <table>
          <thead><tr><th>Steps</th><th>Milestone</th><th>Eval success</th><th>Eval mean reward</th></tr></thead>
          <tbody>
            {checkpoints.map((item) => (
              <tr key={item.id}>
                <td className="mono">{item.timesteps}</td>
                <td>{item.metrics?.milestone ?? "—"}</td>
                <td>{item.metrics?.evaluation ? Number(item.metrics.evaluation.success_rate).toFixed(2) : "—"}</td>
                <td>{item.metrics?.evaluation ? Number(item.metrics.evaluation.mean_reward).toFixed(1) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

function AnalyticsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState("");
  const [data, setData] = useState<Analytics | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let stop = false;
    function load() {
      api<Experiment[]>("/api/v1/experiments").then((rows) => {
        if (stop) return;
        setExperiments(rows);
        setSelected((current) => current || rows[0]?.id || "");
      }).catch((exc: Error) => { if (!stop) setError(exc.message); });
    }
    load();
    const timer = window.setInterval(load, 2000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, []);
  const experiment = experiments.find((item) => item.id === selected);
  useEffect(() => {
    if (!selected) return;
    let stop = false;
    function load() {
      api<Analytics>(`/api/v1/experiments/${selected}/analytics`).then((body) => {
        if (!stop) setData(body);
      }).catch((exc: Error) => { if (!stop) setError(exc.message); });
    }
    load();
    const active = ["running", "pause_requested", "stop_requested"].includes(experiment?.status || "");
    if (!active) return () => { stop = true; };
    const timer = window.setInterval(load, 2000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [selected, experiment?.status]);
  return (
    <>
      <header className="top">
        <div>
          <h1>Analytics</h1>
          <p>Charts read stored training episodes. An empty chart means that run has no logged episodes yet.</p>
        </div>
        <select value={selected} onChange={(event) => setSelected(event.target.value)}>
          {experiments.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
      </header>
      {error && <p className="error">{error}</p>}
      <div className="grid cards">
        <Card title="Steps" value={String(data?.timesteps_completed ?? 0)} />
        <Card title="Episodes" value={String(data?.episodes.length ?? 0)} />
        <Card title="Landings in training" value={String(data?.landing_count ?? 0)} />
        <Card title="Mean landing speed" value={data?.mean_landing_speed_mps == null ? "no landings" : data.mean_landing_speed_mps.toFixed(2) + " m/s"} />
        <Card title="Mean duration" value={data?.mean_episode_length == null ? "—" : data.mean_episode_length.toFixed(0) + " steps"} />
        <Card title="Mean final distance" value={data?.mean_final_distance_m == null ? "—" : data.mean_final_distance_m.toFixed(2) + " m"} />
      </div>
      {!data?.episodes.length && <p className="panel">No training episodes have been recorded for {experiment?.name || "this experiment"}.</p>}
      {!!data?.episodes.length && (
        <div className="grid two" style={{ marginTop: 14 }}>
          <Chart title="Episode reward" data={data.episodes.map((row) => ({ timestep: row.timestep, value: row.episode_reward }))} />
          <Chart title="Rolling average reward" data={data.rolling_reward} />
          <Chart title="Landing rate (20-episode window)" data={data.success_rate} />
          <Chart title="Crash rate (20-episode window)" data={data.crash_rate} />
        </div>
      )}
    </>
  );
}

function Replay() {
  const [rows, setRows] = useState<TrajectorySummary[]>([]);
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [left, setLeft] = useState<Trajectory | null>(null);
  const [right, setRight] = useState<Trajectory | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  useEffect(() => {
    api<TrajectorySummary[]>("/api/v1/trajectories").then((items) => {
      setRows(items);
      setLeftId(items[0]?.id || "");
    }).catch(() => setRows([]));
  }, []);
  useEffect(() => {
    if (!leftId) return;
    api<Trajectory>(`/api/v1/trajectories/${leftId}`).then((payload) => {
      setLeft(payload);
      setIndex(0);
    }).catch(() => setLeft(null));
  }, [leftId]);
  useEffect(() => {
    if (!rightId) {
      setRight(null);
      return;
    }
    api<Trajectory>(`/api/v1/trajectories/${rightId}`).then(setRight).catch(() => setRight(null));
  }, [rightId]);
  useEffect(() => {
    if (!playing || !left) return;
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= left.steps.length - 1) return current;
        return current + 1;
      });
    }, 80 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, left]);
  const step = left?.steps[index];
  const compareStep = right?.steps[Math.min(index, (right?.steps.length || 1) - 1)];
  return (
    <>
      <header className="top">
        <div>
          <h1>Flight replay</h1>
          <p>Playback uses the recorded state, action and reward. It does not resimulate a new landing.</p>
        </div>
      </header>
      <div className="row" style={{ marginBottom: 12 }}>
        <label>Episode<select value={leftId} onChange={(event) => setLeftId(event.target.value)}>{rows.map((row) => <option key={row.id} value={row.id}>{row.label} · {row.outcome}</option>)}</select></label>
        <label>Compare<select value={rightId} onChange={(event) => setRightId(event.target.value)}><option value="">None</option>{rows.map((row) => <option key={row.id} value={row.id}>{row.label} · {row.outcome}</option>)}</select></label>
        <button className="btn" onClick={() => setPlaying((value) => !value)} disabled={!left}>{playing ? "Pause" : "Play"}</button>
        {[0.5, 1, 2, 4].map((value) => <button key={value} className="btn" onClick={() => setSpeed(value)}>{value}×</button>)}
        {leftId && <a className="btn" href={`/api/v1/trajectories/${leftId}/export.csv`}>CSV</a>}
        {leftId && <a className="btn" href={`/api/v1/trajectories/${leftId}/export.json`}>JSON</a>}
      </div>
      {!left && <p className="panel">No recorded flights yet. Finish a simulator episode or an evaluation to create one.</p>}
      {left && step && (
        <div className="grid two">
          <div>
            <RocketView terrain={left.terrain || []} pad={left.pad || { x: 0, half_width: 4 }} geometry={step.scene || null} trail={left.steps.slice(0, index + 1).map((item) => [Number(item.raw.x), Number(item.raw.y)])} />
            <input className="scrub" type="range" min={0} max={left.steps.length - 1} value={index} onChange={(event) => setIndex(Number(event.target.value))} />
          </div>
          <section className="panel">
            <h2>{left.label} · {left.outcome}</h2>
            <div className="telemetry">
              <Stat label="Step" value={String(step.step)} />
              <Stat label="Action" value={step.action_name || "—"} />
              <Stat label="Reward" value={fmt(step.reward)} />
              <Stat label="Cumulative" value={fmt(step.cumulative_reward)} />
              <Stat label="Distance" value={fmt(step.distance_m, "m")} />
              <Stat label="Speed" value={fmt(step.speed_mps, "m/s")} />
            </div>
            <RewardList parts={step.reward_parts} />
            <p className="mono">obs {step.observation.map((value) => value.toFixed(3)).join("  ")}</p>
            {compareStep && <p className="note">Comparison episode is at reward {fmt(compareStep.cumulative_reward)} with outcome {compareStep.outcome || right?.outcome}.</p>}
          </section>
        </div>
      )}
    </>
  );
}

function Compare() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [checkpointId, setCheckpointId] = useState("");
  const [includeRandom, setIncludeRandom] = useState(true);
  const [includeLlm, setIncludeLlm] = useState(false);
  const [group, setGroup] = useState("");
  const [status, setStatus] = useState("idle");
  const [results, setResults] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    api<Checkpoint[]>("/api/v1/checkpoints").then((rows) => {
      setCheckpoints(rows);
      setCheckpointId(rows[0]?.id || "");
    }).catch(() => setCheckpoints([]));
  }, []);
  const comparisonBusy = status === "running" || results.some((row) => row.status === "running");
  useEffect(() => {
    if (!group || !comparisonBusy) return;
    const timer = window.setInterval(() => {
      api<{ status: string; results: Record<string, unknown>[] }>(`/api/v1/comparisons/${group}`).then((body) => {
        setStatus(body.status);
        setResults(body.results);
      }).catch((exc: Error) => {
        setError(exc.message);
        setStatus("failed");
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [group, comparisonBusy]);
  function run() {
    setError("");
    const controllers: { type: string; checkpoint_id?: string }[] = [];
    if (includeRandom) controllers.push({ type: "random" });
    if (checkpointId) controllers.push({ type: "ppo", checkpoint_id: checkpointId });
    if (includeLlm) controllers.push({ type: "llm" });
    if (!controllers.length) {
      setError("Choose at least one controller.");
      return;
    }
    setResults([]);
    setGroup("");
    setStatus("running");
    api<{ group_id: string }>("/api/v1/comparisons", {
      method: "POST",
      body: JSON.stringify({ seeds: [0, 1, 2], episodes_per_seed: 1, controllers }),
    }).then((body) => setGroup(body.group_id)).catch((exc: Error) => {
      setStatus("idle");
      setError(exc.message);
    });
  }
  return (
    <>
      <header className="top">
        <div>
          <h1>Experiment comparison</h1>
          <p>Every controller below is rolled out on the same seeds and the default physics. Human flights stay in the replay list because they are not batched.</p>
        </div>
        <span className={`pill ${status}`}>{status}</span>
      </header>
      {error && <p className="error">{error}</p>}
      <div className="row" style={{ marginBottom: 12 }}>
        <label><input type="checkbox" checked={includeRandom} onChange={(event) => setIncludeRandom(event.target.checked)} /> Random</label>
        <label>
          PPO
          <select value={checkpointId} onChange={(event) => setCheckpointId(event.target.value)}>
            <option value="">None</option>
            {checkpoints.map((item) => <option key={item.id} value={item.id}>{(item.experiment_name || "run") + " · " + item.timesteps}</option>)}
          </select>
        </label>
        <label><input type="checkbox" checked={includeLlm} onChange={(event) => setIncludeLlm(event.target.checked)} /> LLM, if configured</label>
        <button className="btn primary" disabled={comparisonBusy} onClick={run}>Run comparison</button>
      </div>
      <section className="panel">
        {!results.length && <p>No comparison has been stored yet.</p>}
        <table>
          <thead><tr><th>Controller</th><th>Status</th><th>Mean reward</th><th>Std</th><th>Landing rate</th><th>Crash rate</th><th>Note</th></tr></thead>
          <tbody>
            {results.map((row) => {
              const metrics = (row.metrics || {}) as Record<string, number | string>;
              return (
                <tr key={String(row.id)}>
                  <td>{String(row.label)}</td>
                  <td>{String(row.status)}</td>
                  <td>{num(metrics.mean_reward)}</td>
                  <td>{num(metrics.std_reward)}</td>
                  <td>{num(metrics.success_rate)}</td>
                  <td>{num(metrics.crash_rate)}</td>
                  <td>{String(metrics.warning || metrics.error || "")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong className="mono">{value}</strong></div>;
}

function RewardList({ parts }: { parts?: Frame["reward_parts"] }) {
  const rows = [
    ["Distance", parts?.distance],
    ["Speed", parts?.speed],
    ["Upright", parts?.upright],
    ["Landing bonus", parts?.landing_bonus],
    ["Crash penalty", parts?.crash_penalty],
  ] as const;
  return (
    <div className="telemetry" style={{ marginTop: 12 }}>
      {rows.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong className={value != null && value < 0 ? "neg" : "pos"}>{value == null ? "—" : value.toFixed(3)}</strong>
        </div>
      ))}
    </div>
  );
}

function Card({ title, value }: { title: string; value: string }) {
  return <section className="panel flat"><h2>{title}</h2><p className="mono" style={{ color: "var(--text)", fontSize: 20 }}>{value}</p></section>;
}

function Chart({ title, data }: { title: string; data: { timestep: number; value: number }[] }) {
  const points = useMemo(() => data, [data]);
  return (
    <section className="panel">
      <h2>{title}</h2>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={points}>
          <CartesianGrid stroke="#243140" />
          <XAxis dataKey="timestep" stroke="#8ea0b3" />
          <YAxis stroke="#8ea0b3" />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke="#79d2ff" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}

function fmt(value: number | undefined, unit = "") {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(2)}${unit ? " " + unit : ""}`;
}

function num(value: number | string | undefined) {
  return typeof value === "number" ? value.toFixed(2) : "—";
}
