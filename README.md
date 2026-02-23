# TEF-DHN-Service-1---Heating-Consumption-Optimisation
Heating Consumption Optimisation is a data-driven service that analyses operational heating network data to detect inefficiencies and optimise system performance, reducing energy consumption and peak demand while maintaining comfort.

One of the biggest challenges associated with DHN optimization is the absence of labeled anomalous consumption data and the difficulty in establishing objective efficiency thresholds across diverse building types and operational conditions. Another challenge is the multimodal nature of heating systems, which operate under different regimes (day/night, weekday/weekend, winter/summer) making simple threshold-based approaches insufficient.

We follow an approach that combines rule-based anomaly detection with baseline analysis to identify heating consumption inefficiencies in District Heating Network time series data. The service describes a methodology that establishes consumption baselines from 5 years of historical data from Torrelago district, detecting anomalies through predefined rules such as:

Potential applications are various in District Heating Networks such as:

Detection of abnormal consumption peaks without climatic justification
Identification of low ΔT (temperature differential) indicating poor heat transfer efficiency
Detection of high return temperatures signaling system inefficiencies
Sensor malfunction identification (e.g., physically impossible readings)
Hydraulic imbalances and flow anomalies
Baseline establishment for consumption optimization and energy savings quantification

The service enables automatic KPI generation to demonstrate value, including anomaly evolution tracking, building/substation rankings by performance, and quantified economic savings potential through efficiency improvements.

