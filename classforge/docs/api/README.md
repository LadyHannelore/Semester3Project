# ClassForge API Documentation

This document provides a detailed description of the ClassForge API endpoints, request formats, and response structures.

## Base URL

All API endpoints are relative to the base URL:

`
http://localhost:5001/api
`

## Endpoints

### Classroom Management

#### Allocate Students to Classrooms

**Endpoint:** /classrooms/allocate

**Method:** POST

**Description:** Allocate students to classrooms using the genetic algorithm optimization.

**Request Format:**

`json
{
  "students": [
    {
      "id": "1001",
      "academicScore": 85.2,
      "wellbeingScore": 3.5,
      "bullyingScore": 2.1,
      "friends": "1005,1008,1012"
    },
    {
      "id": "1002",
      "academicScore": 72.5,
      "wellbeingScore": 4.0,
      "bullyingScore": 1.3,
      "friends": "1007,1009"
    }
    // ... more students
  ],
  "params": {
    "maxClassSize": 25,
    "maxBulliesPerClass": 2,
    "wellbeingMin": 3.0,
    "wellbeingMax": null,
    "generations": 50,
    "populationSize": 100
  }
}
`

**Response Format:**

`json
{
  "success": true,
  "metrics": {
    "totalStudents": 100,
    "numClasses": 4,
    "avgAcademic": 72.3,
    "avgWellbeing": 3.5,
    "balanceScore": 0.85,
    "diversityScore": 0.92,
    "constraintSatisfaction": 1.0,
    "processingTime": "5.2s"
  },
  "violations": [],
  "classes": [
    {
      "classId": 0,
      "students": [
        {
          "id": "1001",
          "academicScore": 85.2,
          "wellbeingScore": 3.5,
          "bullyingScore": 2.1,
          "friends": "1005,1008,1012"
        },
        // ... more students
      ]
    },
    // ... more classes
  ]
}
`

**Error Response:**

`json
{
  "success": false,
  "error": "Error message details"
}
`

#### Generate Synthetic Data

**Endpoint:** /classrooms/generate

**Method:** POST

**Description:** Generate synthetic student data for testing.

**Request Format:**

`json
{
  "numStudents": 1000
}
`

**Response Format:**

`json
{
  "success": true,
  "students": [
    {
      "id": "1001",
      "academicScore": 85.2,
      "wellbeingScore": 3.5,
      "bullyingScore": 2.1,
      "friends": "1005,1008,1012"
    },
    // ... more students
  ]
}
`

## Error Codes

- 400 - Bad Request: The request is malformed or missing required data.
- 500 - Internal Server Error: Something went wrong on the server side.

## Data Models

### Student

- id (string): Unique student identifier
- cademicScore (float): Academic performance score (0-100)
- wellbeingScore (float): Wellbeing assessment score (1-5)
- ullyingScore (float): Bullying risk score (0-10)
- riends (string): Comma-separated list of student IDs representing friendships

### Allocation Parameters

- maxClassSize (integer): Maximum number of students per class
- maxBulliesPerClass (integer): Maximum number of high-risk bullies per class
- wellbeingMin (float, optional): Minimum average wellbeing score per class
- wellbeingMax (float, optional): Maximum average wellbeing score per class
- generations (integer): Number of generations for genetic algorithm
- populationSize (integer): Population size for genetic algorithm
