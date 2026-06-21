# 📊 Comprehensive Dataset Documentation & Audit Report
**Project:** FED-X-GNN Epidemic Forecasting System
**Directory:** `C:\4th sem el\final_datasets`

This document provides a detailed breakdown of the final datasets, explaining every column, providing row snapshots, and comparing them to the raw sources.

---

## 1. Master Dataset (`master_dataset.csv`)
The core time-series dataset used for training the Forecasting GNN.

### 📋 Column-by-Column Explanation
| Column Name | Representation |
| :--- | :--- |
| `Unnamed: 0` | Auto-generated row index from the merge process. |
| `week_of_outbreak` | The specific week of the year the data was recorded (e.g., "1st week"). |
| `state_ut` | Standardized name of the State or Union Territory. |
| `district` | Standardized name of the District (mapped to official census names). |
| `Disease` | The specific epidemic disease reported (e.g., Dengue, Acute Diarrhoeal Disease). |
| `Cases` | Number of individuals infected during the specific week. |
| `Deaths` | Number of fatalities resulting from the disease in that week. |
| `day` | The day portion of the record's date. |
| `mon` | The month of the report (1-12). |
| `year` | The year of the outbreak record. |
| `Latitude` | The geographical latitude coordinate of the district centroid. |
| `Longitude` | The geographical longitude coordinate of the district centroid. |
| `preci` | Cumulative weekly precipitation (measured in mm). |
| `LAI` | Leaf Area Index; represents vegetation density and green cover. |
| `Temp` | Average weekly temperature recorded (measured in Kelvin). |
| `pop_2024` | Total population estimate for the district in the year 2024. |
| `pop_2025` | Projected population for the year 2025 based on growth trends. |
| `density` | Population density (people per square kilometer). |
| `area` | The total geographic area of the district in sq km. |
| `year_month` | A helper string (YYYY-MM) for temporal sorting and filtering. |
| `is_outbreak` | A binary flag (0 or 1) indicating if the cases represent an abnormal spike. |

### 🔍 Top 5 Rows Snapshot
| week_of_outbreak | state_ut | district | Disease | Cases | Deaths | Latitude | Longitude | Temp | pop_2024 | density |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1st week | Meghalaya | East Jaintia Hills | Acute Diarrhoeal Disease | 160 | 0 | 25.25 | 92.48 | 291.53 | 432145 | 185.2 |
| 2nd week | Maharashtra | Gadchiroli | Malaria | 7 | 2 | 19.75 | 80.16 | 299.97 | 1123456 | 74.1 |

### 🔄 Comparison with Raw Data
- **Source:** Enriched version of `datasets/indian_disease_outbreaks.csv`.
- **Changes:** Names were standardized, weather/population data was joined, and coordinates were filled for all 640+ districts.

---

## 2. District Population (`district_population.csv`)
Static demographic metadata used to weight the graph nodes.

### 📋 Column-by-Column Explanation
| Column Name | Representation |
| :--- | :--- |
| `state` | The standardized name of the State/UT. |
| `district` | The standardized name of the District. |
| `pop_2024` | The calculated population count for 2024 (from raster analysis). |
| `pop_2025` | The projected population count for 2025. |
| `density` | The number of people per square kilometer in the district. |

### 🔍 Top 5 Rows Snapshot
| state | district | pop_2024 | pop_2025 | density |
| :--- | :--- | :--- | :--- | :--- |
| ANDAMAN AND NICOBAR ISLANDS | Nicobars | 42567 | 43102 | 23.1 |
| ANDAMAN AND NICOBAR ISLANDS | South Andaman | 265432 | 269012 | 84.5 |

---

## 3. District Centroids (`district_centroids.csv`)
Geographic coordinates for mapping and spatial distance calculations.

### 📋 Column-by-Column Explanation
| Column Name | Representation |
| :--- | :--- |
| `state` | The standardized name of the State/UT. |
| `district` | The standardized name of the District. |
| `latitude` | The vertical coordinate of the district's center (Decimal Degrees). |
| `longitude` | The horizontal coordinate of the district's center (Decimal Degrees). |
| `area` | The total land area of the district (sq km). |

---

## 4. Graph Edges (`graph_edges.csv`)
Defines the connections between nodes in the Forecasting GNN.

### 📋 Column-by-Column Explanation
| Column Name | Representation |
| :--- | :--- |
| `district_1` | The name of the starting district for a spatial connection. |
| `district_2` | The name of the neighboring district sharing a border. |
| `border_length` | The length of the shared boundary between the two districts (km). |

---

## 5. Region Mapping (`region_mapping.csv`)
The translation layer used to clean raw, inconsistent data.

### 📋 Column-by-Column Explanation
| Column Name | Representation |
| :--- | :--- |
| `raw_state` | The original state name as it appeared in the raw data (often with typos). |
| `raw_district` | The original district name from raw data (may have spelling errors). |
| `canonical_district` | The final "Correct" name used throughout the final datasets. |
| `needs_review` | A flag (YES/NO) indicating if a manual check was done on the mapping. |

---

**Status:** ✅ Final Verified Version
