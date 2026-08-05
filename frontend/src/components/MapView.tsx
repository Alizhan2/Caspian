import "leaflet/dist/leaflet.css";
import { useEffect } from "react";
import {
  GeoJSON,
  ImageOverlay,
  MapContainer,
  Rectangle,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";
import type { LatLngBoundsExpression, LeafletMouseEvent } from "leaflet";
import type { Detection, Scene } from "../types";

const CASPIAN_CENTER: [number, number] = [42.7, 51.7];
const CASPIAN_BOUNDS: LatLngBoundsExpression = [[36.5, 46.5], [47.5, 56.5]];

function MapResize() {
  const map = useMap();
  useEffect(() => {
    const timer = window.setTimeout(() => map.invalidateSize(), 80);
    return () => window.clearTimeout(timer);
  }, [map]);
  return null;
}

function SelectionHandler({ drawing, onSelect }: { drawing: boolean; onSelect: (point: [number, number]) => void }) {
  useMapEvents({
    click: (event: LeafletMouseEvent) => {
      if (drawing) onSelect([event.latlng.lat, event.latlng.lng]);
    },
  });
  return null;
}

export function MapView({
  detections,
  scene,
  selectedPoint,
  drawing,
  showSatellite,
  showDetections,
  layerOpacity,
  onSelect,
}: {
  detections: Detection[];
  scene: Scene | null;
  selectedPoint: [number, number] | null;
  drawing: boolean;
  showSatellite: boolean;
  showDetections: boolean;
  layerOpacity: number;
  onSelect: (point: [number, number]) => void;
}) {
  const selectionBounds: LatLngBoundsExpression | null = selectedPoint
    ? [[selectedPoint[0] - 0.16, selectedPoint[1] - 0.34], [selectedPoint[0] + 0.16, selectedPoint[1] + 0.34]]
    : null;
  const sceneBounds: LatLngBoundsExpression | null = scene?.bbox?.length === 4
    ? [[scene.bbox[1], scene.bbox[0]], [scene.bbox[3], scene.bbox[2]]]
    : null;

  return (
    <MapContainer center={CASPIAN_CENTER} zoom={6} minZoom={5} maxBounds={CASPIAN_BOUNDS} className="guardian-map" zoomControl>
      <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <MapResize />
      <SelectionHandler drawing={drawing} onSelect={onSelect} />
      {showSatellite && scene?.preview_url && sceneBounds ? (
        <ImageOverlay url={scene.preview_url} bounds={sceneBounds} opacity={layerOpacity / 100} zIndex={300} />
      ) : null}
      {selectionBounds ? (
        <Rectangle bounds={selectionBounds} pathOptions={{ color: "#56f4e9", weight: 1.5, dashArray: "6 5", fillColor: "#56f4e9", fillOpacity: 0.08 }} />
      ) : null}
      {showDetections ? detections.map((detection) => (
        <GeoJSON
          key={`${detection.id}-${layerOpacity}`}
          data={detection.geometry}
          pathOptions={{
            color: detection.risk_level === "HIGH" ? "#ff526b" : detection.risk_level === "MEDIUM" ? "#ffbd58" : "#60e6a0",
            weight: 2,
            opacity: Math.max(0.45, layerOpacity / 100),
            fillOpacity: (layerOpacity / 100) * 0.55,
          }}
        />
      )) : null}
    </MapContainer>
  );
}
