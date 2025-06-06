# Student and Class models for ClassForge
class Student:
    """
    Model representing a student in the system
    """
    def __init__(self, student_id, academic_score=0, wellbeing_score=0, bullying_score=0, friends=None):
        self.id = str(student_id)
        self.academic_score = float(academic_score)
        self.wellbeing_score = float(wellbeing_score)
        self.bullying_score = float(bullying_score)
        self.friends = friends or []
        
    def to_dict(self):
        """Convert student to dictionary representation"""
        return {
            'id': self.id,
            'academicScore': self.academic_score,
            'wellbeingScore': self.wellbeing_score,
            'bullyingScore': self.bullying_score,
            'friends': ','.join(self.friends) if isinstance(self.friends, list) else self.friends
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create student from dictionary"""
        friends = data.get('friends', '')
        if isinstance(friends, str):
            friends = [f.strip() for f in friends.split(',')] if friends.strip() else []
        
        return cls(
            student_id=data['id'],
            academic_score=data.get('academicScore', data.get('Academic_Performance', 0)),
            wellbeing_score=data.get('wellbeingScore', data.get('Wellbeing_Score', 0)),
            bullying_score=data.get('bullyingScore', data.get('Bullying_Score', 0)),
            friends=friends
        )

class Classroom:
    """
    Model representing a classroom with students
    """
    def __init__(self, class_id, students=None):
        self.id = class_id
        self.students = students or []
        
    def add_student(self, student):
        """Add a student to the classroom"""
        self.students.append(student)
        
    def remove_student(self, student_id):
        """Remove a student from the classroom"""
        self.students = [s for s in self.students if s.id != student_id]
        
    def get_size(self):
        """Get the number of students in the classroom"""
        return len(self.students)
        
    def get_average_academic(self):
        """Get the average academic score of the classroom"""
        if not self.students:
            return 0
        return sum(s.academic_score for s in self.students) / len(self.students)
        
    def get_average_wellbeing(self):
        """Get the average wellbeing score of the classroom"""
        if not self.students:
            return 0
        return sum(s.wellbeing_score for s in self.students) / len(self.students)
        
    def count_high_risk_bullies(self):
        """Count the number of high-risk bullies in the classroom"""
        return sum(1 for s in self.students if s.bullying_score > 7)
        
    def to_dict(self):
        """Convert classroom to dictionary representation"""
        return {
            'classId': self.id,
            'students': [s.to_dict() for s in self.students]
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create classroom from dictionary"""
        students = [Student.from_dict(s) for s in data.get('students', [])]
        return cls(class_id=data['classId'], students=students)
