Use University;

SELECT name, course_id
FROM student, takes
WHERE student.ID=takes.ID;

SELECT name, course_id FROM student
NATURAL JOIN takes;

SELECT name, course_id, prereq_id FROM student
NATURAL JOIN takes NATURAL JOIN prereq
WHERE student.ID=takes.ID AND
	takes.course_id=prereq.course_id;
    
SELECT * FROM student NATURAL JOIN takes, course ORDER BY ID;

SELECT * FROM (SELECT * FROM student NATURAL JOIN takes) 
AS A, course ORDER BY ID;

SELECT * FROM student NATURAL JOIN takes NATURAL JOIN course;

SELECT * FROM (SELECT * FROM student NATURAL JOIN takes) 
AS A NATURAL JOIN course;

SELECT name, title
FROM student NATURAL JOIN takes, course
WHERE takes.course_id=course.course_id;

SELECT name, title
FROM student NATURAL JOIN takes NATURAL JOIN course;

SELECT * FROM student INNER JOIN takes
ORDER BY student.ID;

SELECT * FROM student NATURAL JOIN takes
ORDER BY student.ID;

SELECT name, course_id FROM student
JOIN takes;

SELECT name, course_id FROM student
INNER JOIN takes;

SELECT name, course_id FROM student
CROSS JOIN takes;

SELECT * FROM student JOIN takes
ON student.ID=takes.ID
ORDER BY student.ID;

SELECT * FROM student NATURAL JOIN takes
ORDER BY student.ID;

SELECT * FROM course NATURAL LEFT OUTER JOIN prereq;

SELECT * FROM course NATURAL RIGHT OUTER JOIN prereq;

# full outer join -> syntax error
SELECT * FROM course NATURAL LEFT OUTER JOIN prereq UNION
SELECT * FROM course NATURAL RIGHT OUTER JOIN prereq;

SELECT * FROM course NATURAL LEFT OUTER JOIN prereq;

SELECT course_id, title, dept_name, credits, prereq_id FROM course NATURAL LEFT JOIN prereq UNION
SELECT course_id, title, dept_name, credits, prereq_id FROM course NATURAL RIGHT JOIN prereq;

SELECT * FROM course INNER JOIN prereq ON
course.course_id=prereq.course_id;

SELECT * FROM course LEFT OUTER JOIN prereq ON
course.course_id=prereq.course_id;

SELECT * FROM course NATURAL RIGHT OUTER JOIN prereq;

CREATE VIEW faculty AS
SELECT ID, name, dept_name FROM instructor;

SELECT name FROM faculty
WHERE dept_name='Biology';

CREATE VIEW departments_total_salary(dept_name, total_salary) AS
SELECT dept_name, SUM(salary)
FROM instructor
GROUP BY dept_name;

CREATE VIEW faculty_teaching AS SELECT ID, name, dept_name, course_id
FROM instructor NATURAL JOIN teaches;

CREATE VIEW physics_fall_2017 AS
SELECT course.course_id, sec_id, building, room_number
FROM course, section
WHERE course.course_id=section.course_id
	and course.dept_name='Physics'
    and section.semester='Fall'
    and section.year='2017';

CREATE VIEW physics_fall_2017_watson AS
SELECT course_id, room_number
FROM physics_fall_2017
WHERE building='Waston';

CREATE VIEW physics_fall_2017_watson AS
SELECT course_id, room_number
FROM physics_fall_2017
WHERE building='Watson';

CREATE VIEW physics_fall_2017_watson AS
SELECT course_id, room_number
FROM (SELECT course.course_id, building, room_number
	FROM course, section
    WHERE course.course_id=section.course_id
		and course.dept_name='Physics'
        and section.semester='Fall'
        and section.year='2017') as A
	WHERE building='Watson';
    
CREATE VIEW faculty AS SELECT ID, name, dept_name FROM instructor;

INSERT INTO faculty VALUES ('30765', 'Green', 'Music');

CREATE VIEW instructor_info AS
SELECT ID, name, building
FROM instructor, department
WHERE instructor.dept_name=department.dept_name;

INSERT INTO instructor_info
VALUES ('69987', 'White', 'Taylor');

CREATE VIEW history_instructors AS
SELECT * FROM instructor
WHERE dept_name='History';

INSERT INTO history_instructors VALUES ('25666', 'Brown', ' Biology', 100000);

CREATE TABLE department (
	dept_name VARCHAR(20) NOT NULL,
    building VARCHAR(15) DEFAULT NULL,
    budgjt DECIMAL(12, 2) DEFAULT NULL,
    PRIMARY KEY (dept_name)
);

CREATE TABLE teaches (
	ID VARCHAR(5) NOT NULL,
    course_id VARCHAR(8) NOT NULL,
    sec_id VARCHAR(8) NOT NULL,
    semester VARCHAR(6) NOT NULL,
    year DECIMAL(4, 0) NOT NULL,
    PRIMARY KEY (ID, course_id, sec_id, semester, year)
);

CREATE TABLE `course` (
	`course_id` varchar(8) NOT NULL,
	`title` varchar(50) DEFAULT NULL,
	`dept_name` varchar(20) DEFAULT NULL,
	`credits` decimal(2,0) DEFAULT NULL,
	PRIMARY KEY (`course_id`),
	KEY `dept_name` (`dept_name`),
	CONSTRAINT `course_ibfk_1`
	FOREIGN KEY (`dept_name`) REFERENCES
	`department` (`dept_name`)
	ON DELETE SET NULL
);

create table section(
	course_id varchar (8),
	sec_id varchar (8),
	semester varchar (6),
	year numeric (4,0),
	building varchar (15),
	room_number varchar (7),
	time_slot_id varchar (4),
	primary key (course_id, sec_id, semester, year),
	check (semester in ('Fall', 'Winter', 'Spring', 'Summer'))
);

CREATE TABLE Persons (
	ID int Not NULL,
    Lastname VARCHAR(255) NOT NULL,
    Firstname VARCHAR(255),
    Age int,
    City VARCHAR(255) DEFAULT '서울'
);

INSERT INTO Persons VALUES (128, '홍', '길동', 24, DEFAULT);

CREATE TABLE person (
    ID CHAR(10),
    name CHAR(40),
    mother CHAR(10),
    father CHAR(10),
    PRIMARY KEY (ID),
    FOREIGN KEY (father) REFERENCES person(name),
    FOREIGN KEY (mother) REFERENCES person(name) 
);

CREATE TABLE student (
	ID varchar (5),
	name varchar (20) not null,
	dept_name varchar (20),
	tot_cred numeric (3,0) default 0,
	primary key (ID)
);

CREATE INDEX studentID_index ON student(ID);
SHOW INDEX FROM student;

DROP INDEX studentID_index on student;
SHOW INDEX FROM student;

CREATE VIEW geo_instructor AS
(select *
from instructor
where dept_name = 'Geology');