import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { LabelSample } from "../types";
import { GlassCard } from "./GlassCard";

type Language = "ru" | "kk";
type Point = [number, number];

const copy = {
  ru: { eyebrow: "ЭКСПЕРТНАЯ РАЗМЕТКА", title: "Подготовка проверенного датасета", intro: "Отмечайте только нефтеподобные аномалии, которые вы проверили по дополнительным данным. Пустая сцена не считается отрицательной, пока специалист явно её не подтвердит.", queue: "Очередь сцен", unreviewed: "Не проверено", positive: "Есть полигон", negative: "Нет аномалии", scene: "Сцена Sentinel-1", draw: "Нажимайте по контуру потенциальной аномалии", newPolygon: "Новый полигон", finish: "Завершить полигон", undo: "Отменить точку", clear: "Очистить", reviewer: "Имя специалиста", reviewerPlaceholder: "Например, эколог-аналитик", note: "Комментарий проверки", notePlaceholder: "Ветер, суда, сравнение с предыдущей сценой…", savePositive: "Сохранить как положительный", saveNegative: "Подтвердить отсутствие", saving: "Сохранение…", saved: "Экспертная оценка сохранена", needReviewer: "Укажите имя проверяющего", needPolygon: "Для положительного примера завершите хотя бы один полигон", loadError: "Не удалось загрузить локальный label pack", date: "Дата", polygons: "Полигонов", warning: "Это разметка для обучения, а не официальное подтверждение разлива." },
  kk: { eyebrow: "САРАПШЫЛЫҚ БЕЛГІЛЕУ", title: "Тексерілген деректер жиынын дайындау", intro: "Қосымша деректермен тексерілген мұнайға ұқсас ауытқуларды ғана белгілеңіз. Маман нақты растағанға дейін бос көрініс теріс мысал болып саналмайды.", queue: "Көріністер кезегі", unreviewed: "Тексерілмеген", positive: "Полигон бар", negative: "Ауытқу жоқ", scene: "Sentinel-1 көрінісі", draw: "Ықтимал ауытқу контуры бойынша нүктелерді басыңыз", newPolygon: "Жаңа полигон", finish: "Полигонды аяқтау", undo: "Нүктені қайтару", clear: "Тазалау", reviewer: "Маманның аты", reviewerPlaceholder: "Мысалы, эколог-талдаушы", note: "Тексеру түсіндірмесі", notePlaceholder: "Жел, кемелер, алдыңғы көрініспен салыстыру…", savePositive: "Оң мысал ретінде сақтау", saveNegative: "Жоқтығын растау", saving: "Сақталуда…", saved: "Сараптамалық баға сақталды", needReviewer: "Тексерушінің атын көрсетіңіз", needPolygon: "Оң мысал үшін кемінде бір полигонды аяқтаңыз", loadError: "Жергілікті label pack жүктелмеді", date: "Күні", polygons: "Полигондар", warning: "Бұл оқытуға арналған белгілеу, мұнай төгілуінің ресми растамасы емес." },
};

function statusLabel(sample: LabelSample, t: typeof copy.ru) {
  if (sample.review_status === "reviewed_positive") return t.positive;
  if (sample.review_status === "reviewed_negative") return t.negative;
  return t.unreviewed;
}

function regionName(sample: LabelSample, language: Language) {
  const names: Record<string, Record<Language, string>> = {
    aktau: { ru: "Побережье Актау", kk: "Ақтау жағалауы" },
    kashagan: { ru: "Кашаган", kk: "Қашаған" },
    atyrau: { ru: "Шельф Атырау", kk: "Атырау қайраңы" },
  };
  return names[sample.region]?.[language] ?? sample.region_name;
}

export function LabelingPage({ language }: { language: Language }) {
  const t = copy[language];
  const [samples, setSamples] = useState<LabelSample[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [polygons, setPolygons] = useState<Point[][]>([]);
  const [current, setCurrent] = useState<Point[]>([]);
  const [reviewer, setReviewer] = useState("");
  const [note, setNote] = useState("");
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);
  const sample = useMemo(() => samples.find((item) => item.sample_id === selectedId) ?? samples[0], [samples, selectedId]);

  useEffect(() => { api.labelSamples().then((items) => { setSamples(items); setSelectedId((value) => value || items[0]?.sample_id || ""); }).catch(() => setMessage(t.loadError)); }, [t.loadError]);
  useEffect(() => { if (!sample) return; setPolygons(sample.polygons.map((polygon) => polygon.slice(0, -1) as Point[])); setCurrent([]); setReviewer(sample.reviewed_by ?? ""); setMessage(""); }, [sample?.sample_id]);

  function addPoint(event: React.MouseEvent<SVGSVGElement>) {
    if (!sample) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const y = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
    const [west, south, east, north] = sample.bbox;
    setCurrent((points) => [...points, [west + x * (east - west), north - y * (north - south)]]);
  }
  function svgPoints(points: Point[]) { if (!sample) return ""; const [west, south, east, north] = sample.bbox; return points.map(([lon, lat]) => `${((lon - west) / (east - west)) * 1000},${((north - lat) / (north - south)) * 1000}`).join(" "); }
  function finishPolygon() { if (current.length < 3) { setMessage(t.needPolygon); return; } setPolygons((items) => [...items, current]); setCurrent([]); setMessage(""); }
  async function save(status: "reviewed_positive" | "reviewed_negative") {
    if (!sample) return;
    if (reviewer.trim().length < 2) { setMessage(t.needReviewer); return; }
    if (status === "reviewed_positive" && !polygons.length) { setMessage(t.needPolygon); return; }
    setSaving(true); setMessage("");
    try {
      const updated = await api.saveLabelReview(sample.sample_id, status, reviewer.trim(), status === "reviewed_positive" ? polygons : [], note);
      setSamples((items) => items.map((item) => item.sample_id === updated.sample_id ? updated : item));
      setMessage(t.saved);
    } catch { setMessage(t.loadError); } finally { setSaving(false); }
  }

  return <section className="labeling-page">
    <div className="labeling-head"><div><div className="eyebrow">{t.eyebrow}</div><h2>{t.title}</h2><p>{t.intro}</p></div><div className="labeling-warning">! {t.warning}</div></div>
    <div className="labeling-layout">
      <GlassCard className="labeling-queue"><div className="eyebrow">{t.queue}</div>{samples.map((item) => <button key={item.sample_id} className={item.sample_id === sample?.sample_id ? "active" : ""} onClick={() => setSelectedId(item.sample_id)}><span><strong>{regionName(item, language)}</strong><small>{new Date(item.acquisition_time).toLocaleDateString(language === "ru" ? "ru-RU" : "kk-KZ")}</small></span><em className={item.review_status}>{statusLabel(item, t)}</em></button>)}</GlassCard>
      <div className="labeling-workspace">
        {sample ? <><GlassCard className="labeling-scene"><div className="labeling-scene-meta"><div><div className="eyebrow">{t.scene}</div><strong>{regionName(sample, language)}</strong></div><span>{t.date}: {new Date(sample.acquisition_time).toLocaleString(language === "ru" ? "ru-RU" : "kk-KZ")} · {t.polygons}: {polygons.length}</span></div><div className="annotation-canvas"><img src={api.assetUrl(sample.preview_url)} alt={regionName(sample, language)} /><svg viewBox="0 0 1000 1000" onClick={addPoint} aria-label={t.draw}>{polygons.map((polygon, index) => <polygon key={index} points={svgPoints(polygon)} />)}{current.length ? <polyline points={svgPoints(current)} /> : null}{current.map((point, index) => { const [x, y] = svgPoints([point]).split(","); return <circle key={index} cx={x} cy={y} r="8" />; })}</svg><div className="annotation-hint">{t.draw}</div></div><div className="annotation-tools"><button className="button button-secondary" onClick={() => setCurrent([])}>{t.newPolygon}</button><button className="button button-primary" onClick={finishPolygon} disabled={current.length < 3}>{t.finish}</button><button className="button button-secondary" onClick={() => setCurrent((points) => points.slice(0, -1))} disabled={!current.length}>{t.undo}</button><button className="button button-secondary" onClick={() => { setPolygons([]); setCurrent([]); }}>{t.clear}</button></div></GlassCard>
        <GlassCard className="label-review-form"><label>{t.reviewer}<input value={reviewer} onChange={(event) => setReviewer(event.target.value)} placeholder={t.reviewerPlaceholder} /></label><label>{t.note}<textarea value={note} onChange={(event) => setNote(event.target.value)} placeholder={t.notePlaceholder} /></label>{message ? <div className="label-message">{message}</div> : null}<div><button className="button button-primary" disabled={saving || !polygons.length} onClick={() => void save("reviewed_positive")}>{saving ? t.saving : t.savePositive}</button><button className="button button-secondary" disabled={saving} onClick={() => void save("reviewed_negative")}>{t.saveNegative}</button></div></GlassCard></> : <GlassCard className="labeling-empty">{message || t.loadError}</GlassCard>}
      </div>
    </div>
  </section>;
}
