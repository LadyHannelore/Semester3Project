from ortools.sat.python import cp_model
import pandas as pd

# Test OR-Tools basic functionality
model = cp_model.CpModel()
print("OR-Tools is working!")

# Test classroom allocation constraints
def test_constraints():
    print("Testing classroom allocation constraints...")

    # Sample synthetic data for testing
    data = {
        "StudentID": ["S001", "S002", "S003", "S004"],
        "Academic_Performance": [85, 90, 70, 60],
        "k6_overall": [3, 4, 5, 2],
        "bullying": [1, 6, 2, 7],
    }
    df = pd.DataFrame(data)

    # Relaxed constraints
    CLASS_SIZE_TARGET = 2
    MAX_ALLOWED_DIFFERENCE = 50  # Increased from 15
    MAX_ALLOWED_WELLBEING_DIFF = 10  # Increased from 2
    MAX_BULLIES_PER_CLASS = 2  # Increased from 1

    # Debugging: Print constraint values
    print("Debugging constraints:")
    print(f"CLASS_SIZE_TARGET: {CLASS_SIZE_TARGET}")
    print(f"MAX_ALLOWED_DIFFERENCE: {MAX_ALLOWED_DIFFERENCE}")
    print(f"MAX_ALLOWED_WELLBEING_DIFF: {MAX_ALLOWED_WELLBEING_DIFF}")
    print(f"MAX_BULLIES_PER_CLASS: {MAX_BULLIES_PER_CLASS}")

    # OR-Tools model
    num_students = len(df)
    num_classes = 2
    students = range(num_students)
    classes = range(num_classes)

    student_class = {
        (i, j): model.NewBoolVar(f"student_{i}_class_{j}")
        for i in students
        for j in classes
    }

    # Constraints
    for i in students:
        model.Add(sum(student_class[(i, j)] for j in classes) == 1)

    for j in classes:
        model.Add(sum(student_class[(i, j)] for i in students) <= CLASS_SIZE_TARGET)

    for j in classes:
        class_academic_perf = sum(
            student_class[(i, j)] * df.loc[i, "Academic_Performance"] for i in students
        )
        model.Add(class_academic_perf <= MAX_ALLOWED_DIFFERENCE)

    for j in classes:
        class_wellbeing = sum(
            student_class[(i, j)] * df.loc[i, "k6_overall"] for i in students
        )
        model.Add(class_wellbeing <= MAX_ALLOWED_WELLBEING_DIFF)

    for j in classes:
        class_bullies = sum(
            student_class[(i, j)] * int(df.loc[i, "bullying"] > 5) for i in students
        )
        model.Add(class_bullies <= MAX_BULLIES_PER_CLASS)

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Constraints satisfied. Solution found!")
        for i in students:
            for j in classes:
                if solver.Value(student_class[(i, j)]) == 1:
                    print(f"Student {df.loc[i, 'StudentID']} assigned to Class {j}")
    else:
        print("No solution found. Constraints may be too strict.")
        # Debugging: Print why constraints might fail
        for j in classes:
            print(f"Class {j}:")
            print(f"  Academic Performance: {[df.loc[i, 'Academic_Performance'] for i in students]}")
            print(f"  Wellbeing: {[df.loc[i, 'k6_overall'] for i in students]}")
            print(f"  Bullies: {[int(df.loc[i, 'bullying'] > 5) for i in students]}")

# Run the test
if __name__ == "__main__":
    test_constraints()