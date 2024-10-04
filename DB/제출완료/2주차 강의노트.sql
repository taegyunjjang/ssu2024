Use university;

select * from instructor
where dept_name='physics';

select * from instructor
where dept_name='physics' and salary > 90000;

select * from department
where dept_name=building;

select ID, name from instructor;

select ID, name, salary from instructor;

select A.name from (select * from instructor
where dept_name='physics') as A;

select name from instructor
where dept_name='physics';

select instructor.ID as instructor_id, name, dept_name, salary, teaches.ID as teaches_id, course_id, sec_id, semester, year
from instructor, teaches
order by instructor.ID;

select instructor.ID as instructor_id, name, dept_name, salary, teaches.ID as teaches_id, course_id, sec_id, semester, year
from instructor
JOIN teaches ON instructor.id=teaches.id
order by instructor.ID;
Use university;

select * from instructor
where dept_name='physics';

select * from instructor
where dept_name='physics' and salary > 90000;

select * from department
where dept_name=building;

select ID, name from instructor;

select ID, name, salary from instructor;

select A.name from (select * from instructor
where dept_name='physics') as A;

select name from instructor
where dept_name='physics';

select instructor.ID as instructor_id, name, dept_name, salary, teaches.ID as teaches_id, course_id, sec_id, semester, year
from instructor, teaches
order by instructor.ID;

select instructor.ID as instructor_id, name, dept_name, salary, teaches.ID as teaches_id, course_id, sec_id, semester, year
from instructor
JOIN teaches ON instructor.id=teaches.id
order by instructor.ID;

select course_id from section where semester='Fall' and year=2017
UNION
select course_id from section where semester='Spring' and year=2018;

select course_id from section where semester='Fall' and year=2017
UNION
select course_id from section where semester='Spring' and year=2018
order by course_id;

select table1.course_id from (select * from section where
semester='Fall' and year=2017) as table1
inner join (select * from section where semester='Spring' and
year=2018) as table2
on table1.course_id=table2.course_id;

select table1.course_id from (select * from section where
semester='Fall' and year=2017) as table1
where table1.course_id in (select course_id from section where
semester='Spring' and year=2018);

select table1.course_id from (select * from section where
	semester='Fall' and year=2017) as table1
where table1.course_id not in (select course_id from section where
	semester='Spring' and year=2018);
    
create or replace view phys_inst as select name from instructor
	where dept_name='physics';
create or replace view music_inst as select name from instructor
	where dept_name='music';
    
Table phys_inst union Table music_inst;

create table instructor(
	ID char(5),
    name varchar(20),
    dept_name varchar(20),
    salary numeric(8, 2));
    
create table instructor(
	ID varchar(5) NOT NULL,
    name varchar(20) NOT NULL,
    dept_name varchar(20) DEFAULT NULL,
    salary decimal(8, 2) DEFAULT NULL,
    PRIMARY KEY (ID),
    KEY dept_name (dept_name),
    CONSTRAINT intructor_ibfk_1 FOREIGN KEY (dept_name) 
		REFERENCES department (dept_name)
		ON DELETE SET NULL
) ENGINE=innoDB DEFAULT CHARSET=utf8;

create table student(
	ID varchar(5),
    name varchar(20) not null,
    dept_name varchar(20),
    tot_cred numeric(3, 0),
    primary key (ID),
    foreign key (dept_name) references department (dept_name));
    
create table takes(
	ID varchar(5),
    course_id varchar(8),
    sec_id varchar(8),
    semester varchar(6),
    uear numeric(4, 0),
    grade varchar(2),
    primary key (ID, course_id, sec_id, semester, year),
    foreign key (ID) references student(ID),
    foreign key (course_id, sec_id, semester, year)
		references section (course_id, sec_id, semester, year));
        
create table course(
	course_id varchar(8),
    title varchar(50),
    dept_name varchar(20),
    credits numeric(2, 0),
    primary key (course_id),
    foreign key (dept_name)
		references department(dept_name));
        
select distinct dept_name from instructor;
select all dept_name from instructor;

select name, course_id
from instructor, teaches
where instructor.ID=teaches.ID;

select name, course_id
from instructor, teaches
where instructor.ID=teaches.ID
	and instructor.dept_name='Art';

select distinct name from instructor
order by name;

select name from instructor
where salary between 90000 and 100000;

select name, course_id from instructor, teaches
where (instructor.ID, dept_name) = (teaches.ID, 'Biology');

(select course_id from section where semester='Fall' and year=2017)
union(select course_id from section where semester='Spring' and year=2018);

select table1.course_id from (select * from section where
semester='Fall' and year=2017 ) as table1
inner join (select * from section where semester='Spring' and
year=2018) as table2
on table1.course_id = table2.course_id;

select table1.course_id from (select * from section where
semester='Fall' and year=2017 ) as table1
where table1.course_id in (select course_id from section where
semester='Spring' and year=2018);

select table1.course_id from (select * from section where
semester='Fall' and year=2017 ) as table1
where table1.course_id not in (select course_id from section where
semester='Spring' and year=2018);

select avg (salary) from instructor
where dept_name='Physics';

select count(distinct ID) from teaches
where semester='Spring' and year=2018;

select count(*) from course;

select dept_name, avg (salary) as avg_salary 
from instructor group by dept_name;

select dept_name, avg (salary) asavg_salary
from instructor
group by dept_name
having avg (salary) > 42000;

select name, dept_name from instructor
where dept_name in
(select dept_name from department
	where budget > (Select max(budget) from department
		where building='Watson'));
        
select distinct course_id from section
where semester ='Fall' and year=2017 and 
	course_id in (select course_id from section
		where semester ='Spring' and year=2018);
        
select distinct name
from instructor
where name not in ('Mozart','Einstein');

select count(distinct ID) from takes
where (course_id, sec_id, semester, year) in 
	(select course_id, sec_id, semester, year from teaches
		where teaches.ID=10101);
        
select distinct T.name from instructor as T, instructor as S
where T.salary > S.salary and S.dept_name ='Biology';

select name from instructor
where salary > some (select salary from instructor
where dept_name='Biology');

select name from instructor
where salary > all (select salary from instructor
where dept_name='Biology');

select course_id from section as S
where semester='Fall' and year=2017 and
	exists (select * from section as T
		where semester='Spring' and year=2018
			and S.course_id = T.course_id);

select table1.course_id from (select * from section where
 semester='Fall' and year=2017 ) as table1
where table1.course_id not in (select course_id from section where
 semester='Spring' and year=2018);
 
select distinct course.course_id from course, section
	where course.course_id=section.course_id
		and section.year=2017;
        
select T.dept_name, T.avg_salary
from (select dept_name, avg (salary) as avg_salary from instructor
	group by dept_name) as T
where T.avg_salary > 42000;

select dept_name, avg_salary
from (select dept_name, avg (salary)
	from instructor
		group by dept_name)
		as dept_avg (dept_name, avg_salary)
where avg_salary > 42000;

with max_budget (value) as (select max(budget) from department)
select department.dept_name from department, max_budget
	where department.budget=max_budget.value;
    
with dept_total (dept_name, value) as
	(select dept_name, sum(salary) from instructor
		group by dept_name),
	dept_total_avg(value) as (select avg(value) from dept_total)
select dept_name from dept_total, dept_total_avg
where dept_total.value > dept_total_avg.value;

select dept_name, (select count(*) from instructor
	where department.dept_name=instructor.dept_name)
	as num_instructors
	from department;

delete from instructor
	where dept_name in (select dept_name from department
		where building='Watson');

insert into course
	values ('CS-437', 'Database Systems', 'Comp. Sci.', 4);

insert into student
	values ('3003', 'Green', 'Finance', null);
    
insert into instructor
	select ID, name, dept_name, 18000
	from student where dept_name='Music' and tot_cred > 30;
    
update instructor
	set salary=salary * 1.05
	where salary < (select avg (salary) from instructor);

update instructor
	set salary = salary * 1.03
	where salary > 100000;
update instructor
	set salary = salary * 1.05
	where salary <= 100000;

update instructor
	set salary = case
		when salary <= 100000 then salary * 1.05
	else salary * 1.03
		end;
        
update student S
	set tot_cred=(select sum(credits)
	from takes, course
		where takes.course_id=course.course_id
			and S.ID= takes.ID and
			takes.grade <> 'F' and
			takes.grade is not null);