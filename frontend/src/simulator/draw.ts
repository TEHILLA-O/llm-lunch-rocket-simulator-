import type { Scene } from "../types";

export type Star = { x: number; y: number; r: number };
export type Particle = { x: number; y: number; vx: number; vy: number; life: number; color: string };

export function makeStars(width: number, height: number): Star[] {
  const stars: Star[] = [];
  let seed = 19;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  for (let i = 0; i < 90; i += 1) {
    stars.push({ x: rand() * width, y: rand() * height * 0.75, r: rand() > 0.8 ? 1.6 : 1 });
  }
  return stars;
}

export function worldToScreen(x: number, y: number, cameraX: number, width: number, height: number, scale: number) {
  return { sx: width / 2 + (x - cameraX) * scale, sy: height * 0.78 - y * scale };
}

export function drawScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  terrain: { x: number; y: number }[],
  pad: { x: number; half_width: number },
  geometry: Scene | null,
  trail: number[][],
  cameraX: number,
  stars: Star[],
  particles: Particle[],
) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070b12";
  ctx.fillRect(0, 0, width, height);
  const altitude = geometry ? geometry.y : 12;
  const scale = Math.min(34, Math.max(14, (height * 0.62) / Math.max(8, altitude + 3)));
  for (const star of stars) {
    ctx.fillStyle = "#d5deea";
    ctx.fillRect(star.x, star.y, star.r, star.r);
  }
  const project = (x: number, y: number) => worldToScreen(x, y, cameraX, width, height, scale);
  ctx.beginPath();
  terrain.forEach((point, index) => {
    const screen = project(point.x, point.y);
    if (index === 0) ctx.moveTo(screen.sx, screen.sy);
    else ctx.lineTo(screen.sx, screen.sy);
  });
  ctx.strokeStyle = "#8d8374";
  ctx.lineWidth = 2;
  ctx.stroke();
  const left = project(pad.x - pad.half_width, 0);
  const right = project(pad.x + pad.half_width, 0);
  ctx.strokeStyle = "#79d2ff";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(left.sx, left.sy);
  ctx.lineTo(right.sx, right.sy);
  ctx.stroke();
  ctx.fillStyle = "#79d2ff";
  ctx.font = "12px IBM Plex Mono, monospace";
  ctx.fillText("PAD", (left.sx + right.sx) / 2 - 14, left.sy + 16);

  ctx.strokeStyle = "rgba(121, 210, 255, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  trail.forEach((point, index) => {
    const screen = project(point[0], point[1]);
    if (index === 0) ctx.moveTo(screen.sx, screen.sy);
    else ctx.lineTo(screen.sx, screen.sy);
  });
  ctx.stroke();

  for (const particle of particles) {
    const screen = project(particle.x, particle.y);
    ctx.globalAlpha = Math.max(0, particle.life);
    ctx.fillStyle = particle.color;
    ctx.fillRect(screen.sx, screen.sy, 2, 2);
  }
  ctx.globalAlpha = 1;

  if (!geometry) return;
  for (const leg of geometry.legs) {
    const hip = project(leg.hip[0], leg.hip[1]);
    const foot = project(leg.foot[0], leg.foot[1]);
    ctx.strokeStyle = leg.contact ? "#3dce97" : "#d7e2ea";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(hip.sx, hip.sy);
    ctx.lineTo(foot.sx, foot.sy);
    ctx.stroke();
  }
  const body = geometry.body.map((point) => project(point[0], point[1]));
  ctx.beginPath();
  body.forEach((point, index) => (index === 0 ? ctx.moveTo(point.sx, point.sy) : ctx.lineTo(point.sx, point.sy)));
  ctx.closePath();
  ctx.fillStyle = geometry.outcome === "crashed" ? "#6a3038" : "#d5dee8";
  ctx.fill();
  ctx.strokeStyle = "#243140";
  ctx.stroke();

  for (const [name, flame] of Object.entries(geometry.flames)) {
    if (!flame.active) continue;
    const origin = project(flame.origin[0], flame.origin[1]);
    const tip = project(
      flame.origin[0] + flame.direction[0] * (name === "main" ? 1.3 : 0.55),
      flame.origin[1] + flame.direction[1] * (name === "main" ? 1.3 : 0.55),
    );
    ctx.strokeStyle = name === "main" ? "#ffb15a" : "#8fd4ff";
    ctx.lineWidth = name === "main" ? 4 : 2;
    ctx.beginPath();
    ctx.moveTo(origin.sx, origin.sy);
    ctx.lineTo(tip.sx, tip.sy);
    ctx.stroke();
    ctx.fillStyle = name === "main" ? "#ffb15a" : "#8fd4ff";
    ctx.fillText(name.toUpperCase(), tip.sx + 4, tip.sy);
  }

  if (geometry.outcome === "landed") {
    const padPoint = project(pad.x, 0.4);
    ctx.strokeStyle = "#3dce97";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(padPoint.sx, padPoint.sy, 22, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (geometry.outcome === "crashed") {
    const impact = project(geometry.x, geometry.y);
    ctx.strokeStyle = "#ff6b78";
    ctx.beginPath();
    ctx.arc(impact.sx, impact.sy, 16, 0, Math.PI * 2);
    ctx.stroke();
  }
}

export function spawnFlameParticles(geometry: Scene, particles: Particle[], burst: string | null) {
  for (const flame of Object.values(geometry.flames)) {
    if (!flame.active) continue;
    particles.push({
      x: flame.origin[0],
      y: flame.origin[1],
      vx: flame.direction[0] * 1.5,
      vy: flame.direction[1] * 1.5,
      life: 1,
      color: "#ffb15a",
    });
  }
  if (burst === "crashed") {
    for (let i = 0; i < 12; i += 1) {
      const angle = (i / 12) * Math.PI * 2;
      particles.push({
        x: geometry.x,
        y: geometry.y,
        vx: Math.cos(angle) * 2,
        vy: Math.sin(angle) * 2,
        life: 1,
        color: "#ff6b78",
      });
    }
  }
}

export function stepParticles(particles: Particle[]) {
  for (const particle of particles) {
    particle.x += particle.vx * 0.05;
    particle.y += particle.vy * 0.05;
    particle.life -= 0.03;
  }
  return particles.filter((particle) => particle.life > 0).slice(-200);
}
