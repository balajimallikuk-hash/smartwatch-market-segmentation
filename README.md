# Smartwatch Market Segmentation

## Project overview

This project analyses smartwatch consumer data to identify distinct market segments and evaluate their potential attractiveness from a marketing perspective.

The analysis combines unsupervised machine learning, statistical testing, segment profiling, and business-oriented evaluation to turn consumer preference data into potential marketing recommendations.

## Business questions

- How many distinct consumer segments can be identified in the sample?
- How do the segments differ in their smartwatch needs and preferences?
- Which segments appear most attractive for marketing purposes?
- How might the segments differ in suitability for potential business partners?
- What can willingness-to-pay analysis contribute to market evaluation?

## Methodology

### 1. Data preparation and exploration
- Load and inspect the segmentation and descriptor data.
- Review descriptive statistics and potential outliers.
- Prepare consumer rating variables for clustering.

### 2. Clustering and model evaluation
- Apply K-Means clustering and Ward hierarchical clustering.
- Compare candidate solutions using silhouette score, Calinski-Harabasz score, and Davies-Bouldin score.
- Examine bootstrap and split-half stability.
- Compare clustering solutions using the Adjusted Rand Index.

### 3. Segment profiling
- Compare consumer needs and preferences across the identified segments.
- Use descriptive summaries and appropriate statistical tests to assess differences.
- Examine categorical descriptors and segment characteristics.

### 4. Business evaluation
- Assess segment attractiveness using a weighted scoring framework.
- Conduct sensitivity analysis to examine how scores change under alternative weighting assumptions.
- Explore potential partner fit and willingness-to-pay patterns.


## Key Findings

### 1. Three consumer segments

The analysis develops a three-segment solution for the smartwatch market:

- **Social Wellness:** A segment to investigate for interest in health, well-being, and related smartwatch features.
- **Always-On Pros:** A segment to investigate for communication, timely information, and productivity-related features.
- **Casual Snappers:** A segment to investigate for photography, style, and creative communication features.

These names are used in the analysis, but their defining characteristics should be confirmed against the calculated segment profiles.

### 2. Data-driven segmentation

The project compares K-Means and Ward hierarchical clustering and evaluates candidate cluster solutions using multiple metrics, including silhouette score, Calinski-Harabasz score, and Davies-Bouldin score.

It also examines bootstrap and split-half stability to investigate whether the segmentation is sensitive to changes in the sample.

### 3. Consumer profiling

The analysis compares the segments across 12 smartwatch attribute-importance ratings and additional respondent characteristics. Statistical tests are used to investigate differences in preferences, willingness to pay, and selected demographic and behavioural descriptors.

### 4. Commercial attractiveness

A weighted scoring framework evaluates the segments using size, willingness to pay, income, consumer needs, strategic fit, and accessibility. Equal-weight and alternative-weight analyses are included to examine how sensitive the rankings are to the chosen assumptions.

### 5. Potential strategic partners

The project compares the potential fit of Google, Amazon, and Aetna with the identified segments, using relevant smartwatch feature preferences and selected ecosystem indicators.

It also examines stated willingness-to-pay patterns to explore potential pricing opportunities.

### Limitations

The findings reflect the available respondent sample and should not automatically be generalised to the entire smartwatch market. Segment names are interpretations of observed patterns, and attractiveness and partner-fit scores depend partly on chosen assumptions and scoring rules. Willingness to pay is not a guarantee of actual purchases or profitability.

## Tools and technologies

- Python
- pandas and NumPy
- scikit-learn
- SciPy
- Matplotlib and Seaborn
- Excel data processing

## Limitations

The findings describe the available respondent sample and should not automatically be generalised to the entire smartwatch market.

Cluster selection and segment attractiveness depend on methodological choices, data quality, and scoring assumptions. The attractiveness scores are decision-support estimates, not objective measures of market success.

## How to run

1. Install the dependencies listed in `requirements.txt`.
2. Place an authorised copy of the required workbook in the expected data folder.
3. Confirm that the input filenames and sheet names match the script.
4. Run:

   `python smartwatch_analysis.py`

5. Review the generated outputs.

## Author

Balaji Mallik

MSc Consumer Analytics and Marketing Strategy  
University of Leeds

