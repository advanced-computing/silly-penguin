# Project 1 Part 1: Proposal
Group [silly-penguin](https://github.com/advanced-computing/silly-penguin): Xingyi Wang, Wuhao Xia

## 1. Dataset
Name: Balancing Authority Areas hourly operating data

Link: [EIA Data Browser - Region Data](https://www.eia.gov/opendata/browser/electricity/rto/region-data)

This dataset provides hourly demand, day-ahead demand forecasts, net generation, and interchange data across 50+ U.S. balancing authorities. It is updated hourly.

## 2. Research Questions
How does the mean absolute percentage error between the "Day-ahead demand forecast" and the "Actual demand" fluctuate during extreme temperature weeks compared to mild weather weeks in certain region, for example the CISO (California) region?

Do balancing authorities differ systematically in their net generation mix and their reliance on power interchange?

Are balancing authorities with insufficient net generation more reliant on interregional power interchange during peak demand periods?

## 3. [Notebook Link](https://github.com/advanced-computing/silly-penguin/blob/main/eia_electricity_project.ipynb)

## 4. Target Visualization

![Time-series line chart comparing forecasted vs. actual load.](https://static.us.edusercontent.com/files/76tHFxrdUKWWM9P8zFtptfYO)

Type: Time-series line chart comparing forecasted vs. actual load.

Description: The plot displays hourly "Day-ahead demand forecast" against "Actual Demand" for the California (CISO) grid.

## 5. Known Unknowns
It is unknown if all 50+ balancing authorities provide a consistent, uninterrupted stream of "Net Generation" breakdown by fuel type for the entire study period.

The exact degree to which non-economic factors, such as local public holidays or unrecorded micro-grid outages, influence "Actual Demand" values remains unquantified.

## 6. Anticipated Challenges
Since the data spans various regions, aligning all period timestamps to a single standard is critical for accurate cross-regional comparison.

Managing and cleaning thousands of rows of hourly data per region can lead to memory bottlenecks in standard Notebook environments.

## 7. Colab Link
<a target="_blank" href="https://colab.research.google.com/github/advanced-computing/silly-penguin">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
