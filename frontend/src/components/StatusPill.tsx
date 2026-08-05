import type { RiskLevel } from "../types";
export function StatusPill({ level, label = level }: { level: RiskLevel; label?: string }) { return <span className={`status-pill status-${level.toLowerCase()}`}><span className="status-dot" />{label}</span>; }
