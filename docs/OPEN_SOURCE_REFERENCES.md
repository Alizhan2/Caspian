# Open-source references

These repositories are references and upstream dependencies, not copied product code.

## Selected repositories

| Repository | Why it matters for Caspian Guardian | Adoption decision |
|---|---|---|
| [sentinel-hub/sentinelhub-py](https://github.com/sentinel-hub/sentinelhub-py) | Official Python interface for Sentinel Hub Catalog and Process APIs, authentication, rate limits, and geospatial utilities. MIT licensed. | Keep the current small HTTP client for the MVP; use this package when batch processing, request splitting, or advanced retry logic is added. |
| [SkyTruth/cerulean-cloud](https://github.com/SkyTruth/cerulean-cloud) | Mature open-source oil-slick monitoring architecture with inference, spatial database, API, and human-in-the-loop validation patterns. Apache-2.0 licensed. | Reuse architectural ideas: immutable review history, machine and human confidence, staged environments, and OGC-compatible outputs. Do not present its model as validated for the Caspian without a separate evaluation. |
| [opengeos/GeoLibre](https://github.com/opengeos/GeoLibre) | Modern React/TypeScript web-GIS built on MapLibre/deck.gl with responsive, local-first geospatial workflows. MIT licensed. | Use as a UX reference for map-first navigation, layer controls, responsive panels, and future high-volume rendering. Keep Leaflet for this MVP to avoid an unnecessary migration. |

## Model policy

The production checkpoint must be open-source, versioned, and evaluated on labelled Sentinel-1 VV/VH examples relevant to the Caspian Sea. A repository containing architecture code alone is not evidence that its weights are scientifically suitable for this region.

Every published detection remains an AI screening signal: **Potential oil-like anomaly requiring field verification.**
