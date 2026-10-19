import { useEffect, useRef } from "react";
import type { Scene } from "../types";
import { drawScene, makeStars, spawnFlameParticles, stepParticles, type Particle, type Star } from "./draw";

type Props = {
  terrain: { x: number; y: number }[];
  pad: { x: number; half_width: number };
  geometry: Scene | null;
  trail: number[][];
};

export function RocketView({ terrain, pad, geometry, trail }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const starsRef = useRef<Star[]>([]);
  const particlesRef = useRef<Particle[]>([]);
  const cameraRef = useRef(0);
  const outcomeRef = useRef<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.floor(rect.width * devicePixelRatio));
    const height = Math.max(280, Math.floor(rect.height * devicePixelRatio));
    canvas.width = width;
    canvas.height = height;
    if (starsRef.current.length === 0) starsRef.current = makeStars(width, height);
    if (geometry) {
      cameraRef.current += (geometry.x - cameraRef.current) * 0.12;
      const burst = geometry.outcome !== outcomeRef.current ? geometry.outcome : null;
      outcomeRef.current = geometry.outcome;
      spawnFlameParticles(geometry, particlesRef.current, burst);
    }
    particlesRef.current = stepParticles(particlesRef.current);
    drawScene(ctx, width, height, terrain, pad, geometry, trail, cameraRef.current, starsRef.current, particlesRef.current);
  }, [terrain, pad, geometry, trail]);

  return (
    <div className="canvas-wrap">
      <canvas ref={canvasRef} aria-label="Rocket simulation" />
    </div>
  );
}
