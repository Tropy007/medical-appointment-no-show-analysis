"""Exploratory analysis of missed medical appointments.

Portfolio project by Oluwatoyin Akadiri.
Expected input file: KaggleV2-May-2016.csv
"""

import pandas as pd
import matplotlib.pyplot as plt


# Load and inspect the appointment data.
df = pd.read_csv("data/KaggleV2-May-2016.csv")

print(df.head())
print("Dataset shape:", df.shape)
print("Column names:", df.columns.tolist())
print("Missing values:")
print(df.isnull().sum())
print("Duplicate records:", df.duplicated().sum())


# Prepare dates and an analysis-friendly weekday field.
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
df["AppointmentWeekday"] = df["AppointmentDay"].dt.day_name()


# Calculate the overall attendance split.
appointment_status = df["No-show"].value_counts()
appointment_percentage = (
    df["No-show"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("Appointment status counts:")
print(appointment_status)
print("Appointment status percentages:")
print(appointment_percentage)


# Visual 1: overall appointment attendance.
attendance = df["No-show"].value_counts()
attendance.index = ["Attended", "Missed"]
attendance.plot(kind="bar", figsize=(7, 5))
plt.title("Patient Appointment Attendance")
plt.xlabel("Appointment Status")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Visual 2: average age by attendance outcome.
age_status = df.groupby("No-show")["Age"].mean()
age_status.index = ["Attended", "Missed"]
age_status.plot(kind="bar", figsize=(7, 5))
plt.title("Average Age by Appointment Status")
plt.xlabel("Appointment Status")
plt.ylabel("Average Age")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Visual 3: appointment attendance by gender.
gender_status = df.groupby("Gender")["No-show"].value_counts().unstack()
gender_status.columns = ["Attended", "Missed"]
gender_status.plot(kind="bar", figsize=(7, 5))
plt.title("Patient Appointment Attendance by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Visual 4: compare patients who did and did not receive an SMS reminder.
sms_status = df.groupby("SMS_received")["No-show"].value_counts().unstack()
sms_status.columns = ["Attended", "Missed"]
sms_status.index = ["No SMS", "Received SMS"]
sms_status.plot(kind="bar", figsize=(7, 5))
plt.title("Appointment Attendance by SMS Reminder")
plt.xlabel("SMS Reminder")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Visual 5: examine scholarship/support status.
scholarship_status = df.groupby("Scholarship")["No-show"].value_counts().unstack()
scholarship_status.columns = ["Attended", "Missed"]
scholarship_status.index = ["No Scholarship", "Has Scholarship"]
scholarship_status.plot(kind="bar", figsize=(7, 5))
plt.title("Appointment Attendance by Scholarship Status")
plt.xlabel("Scholarship Status")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Visual 6: identify the weekdays with the most missed appointments.
missed_appointments = df[df["No-show"] == "Yes"]
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]
weekday_missed = (
    missed_appointments["AppointmentWeekday"]
    .value_counts()
    .reindex(weekday_order)
)
weekday_missed.plot(kind="bar", figsize=(7, 5))
plt.title("Missed Appointments by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Number of Missed Appointments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
