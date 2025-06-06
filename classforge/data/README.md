# ClassForge Data Documentation

This directory contains sample data files and information about the data structure used within ClassForge.

## Data Files

- synthetic_student_data.csv: Sample dataset with 100 synthetic student records
- synthetic_student_data_1000.csv: Larger sample dataset with 1000 synthetic student records

## Data Schema

### Student Data

| Column Name           | Description                                       | Data Type | Range       |
|-----------------------|---------------------------------------------------|-----------|-------------|
| StudentID             | Unique student identifier                         | String    | -           |
| Academic_Performance  | Academic performance score                        | Float     | 0-100       |
| Wellbeing_Score       | Student wellbeing assessment                      | Float     | 1-5         |
| Bullying_Score        | Risk score for bullying behavior                  | Float     | 0-10        |
| Friends               | Comma-separated list of friend StudentIDs         | String    | -           |

### Class Assignment Data

Each student will have a class assignment after allocation:

| Column Name           | Description                                       | Data Type | Range       |
|-----------------------|---------------------------------------------------|-----------|-------------|
| StudentID             | Unique student identifier                         | String    | -           |
| Class_GA              | Class assignment from Genetic Algorithm           | Integer   | 0-n         |

## Data Generation

The synthetic data is generated with the following characteristics:

1. **Academic Performance**: Normal distribution with mean 70 and standard deviation 15
2. **Wellbeing Score**: Normal distribution with mean 3.5 and standard deviation 0.8
3. **Bullying Score**: Exponential distribution with lambda 1.5, capped at 10
4. **Friendship Networks**: Poisson distribution with lambda 3 for number of friends per student

## Working with the Data

To use your own data with ClassForge:

1. Ensure your CSV file follows the schema described above
2. Upload the file using the 'Upload & Simulate' page
3. Verify that your data was loaded correctly by checking the student count

## Data Privacy Considerations

When using real student data, remember to:
- Remove any personally identifiable information beyond the necessary fields
- Use pseudonymized StudentIDs
- Ensure data is stored securely
- Process data in accordance with relevant privacy regulations
