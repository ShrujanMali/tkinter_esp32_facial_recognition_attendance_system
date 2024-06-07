import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime, timedelta
import glob
from sklearn.neighbors import KNeighborsClassifier
import joblib
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
# import openpyxl

if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if f'Attendance_sheet.xlsx' not in os.listdir('Attendance'):
    columns = ['No.', 'Surname', 'Name', 'M/F', 'Standard']
    # Create a DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)
    # Save the DataFrame to an Excel file
    df.to_excel('Attendance/Attendance_sheet.xlsx', index=False)

attendance_sheet = r'Attendance/Attendance_sheet.xlsx'
url = 'http://192.168.1.101/cam-hi.jpg'    # bothesp32
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nimgs = 10

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def validate_fields(characters):
    if characters.isalpha():
        return True
    return False



def add_attendance(name):
        # Read existing data from Excel sheet
        df = pd.read_excel(attendance_sheet)
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d 00:00:00')
        # Add the current date column if it doesn't exist
        if date_string not in df.columns:
            df[date_string] = ""

        # Find the student in the dataframe and mark the attendance
        for index, row in df.iloc[:].iterrows():
            student_name = f"{row['Name']} {row['Surname']}"
            if student_name.upper() == name.upper():
                df.at[index, date_string] = "Present"
                print(f"Attendance marked for {name} on {date_string}")
                break
        else:
            print(f"Student '{name}' not found in the attendance sheet.")
        df.to_excel(attendance_sheet, index=False)


def add_student():
    def handle_gender_selection(event):
        selected_gender = gender_combobox.get()
        if selected_gender == "Select from list":
            submit_button["state"] = tk.DISABLED
        else:
            submit_button["state"] = tk.NORMAL



    def submit_student():
        name = entry_first_name.get()
           # Replace with actual URL of the camera

        if not validate_fields(name):
            print("validating......")
            messagebox.showerror("Invalid Input", 
                                "Name must contain only alphabets.")

        if entry_first_name.get() and entry_surname.get() and entry_standard.get() and gender_combobox.get():
            first_name = entry_first_name.get().title().strip()
            surname = entry_surname.get().title().strip()
            standard = entry_standard.get().upper().strip()
            gender = gender_combobox.get().title().strip()
            print("info", first_name + " " + surname )
            df = pd.read_excel(attendance_sheet)
            for index, row in df.iloc[:].iterrows():
                if (row['Name'] == first_name) and (row['Surname'] == surname):
                    messagebox.showinfo("Error", "Student is already in the list")
                    return
            else:
                messagebox.showinfo("Info", f"Starting camera for capturing photos of student '{first_name} {surname}'.")
                name_drive = 'static/faces/' + first_name + " " + surname
                if not os.path.isdir(name_drive):
                    os.makedirs(name_drive)
                i, j = 0, 0
                nimgs = 10  # Number of images to capture
                while True:
                    img_resp = urllib.request.urlopen(url)
                    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                    frame = cv2.imdecode(imgnp, -1)

                    faces = extract_faces(frame)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                        cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                        if j % 5 == 0:
                            name = first_name + "_" + surname + '_' + str(i) + '.jpg'
                            cv2.imwrite(name_drive + '/' + name, frame[y:y+h, x:x+w])
                            i += 1
                        j += 1
                    if j == nimgs * 5:
                        cv2.destroyAllWindows()
                        max_no = df['No.'].max() if not df.empty else 0
                        new_student_data = {'No.': max_no + 1, 'Surname': surname, 'Name': first_name, 'M/F': gender, 'Standard': standard}
                        df = df._append(new_student_data, ignore_index=True)
                        df.to_excel(attendance_sheet, index=False)
                        train_model()
                        messagebox.showinfo("Success", "Student added successfully")
                        add_student_window.destroy()
                        break
                    cv2.imshow('Adding new User', frame)
                    if cv2.waitKey(1) == 27:  # Press 'ESC' to break
                        break                
                return
        else:
            messagebox.showwarning("Warning", "Please fill all the fields")
            return

            

    add_student_window = tk.Toplevel(root)
    add_student_window.title("Add New Student")
    add_student_window.geometry("300x300")

    tk.Label(add_student_window, text="First Name").pack(pady=5)
    entry_first_name = tk.Entry(add_student_window)
    entry_first_name.pack(pady=5)

    tk.Label(add_student_window, text="Surname").pack(pady=5)
    entry_surname = tk.Entry(add_student_window)
    entry_surname.pack(pady=5)

    tk.Label(add_student_window, text="Standard").pack(pady=5)
    entry_standard = tk.Entry(add_student_window)
    entry_standard.pack(pady=5)


    #######################################################################################################
    gender_options = ["Select from list", "Male", "Female"]
    gender_combobox = ttk.Combobox(add_student_window, values=gender_options, state="readonly")
    gender_combobox.current(0)  # Set default value to "Select from list"
    gender_combobox.pack(pady=10)

    # Bind the selection event to the handle_gender_selection function
    gender_combobox.bind("<<ComboboxSelected>>", handle_gender_selection)

    def handle_submit():
        selected_gender = gender_combobox.get()
        if selected_gender == "Select from list":
            # Display error message (replace with actual form prevention logic)
            messagebox.showerror("Error", "Please select a gender from the list.")
        else:
            print("Form submitted with gender:", selected_gender)  # Simulate submission

    # Create submit button
    submit_button = tk.Button(add_student_window, text="Submit", command=submit_student)
    submit_button.pack(pady=10)
    # Initially disable the submit button
    submit_button["state"] = tk.DISABLED


def mark_attendance():

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print('There is no trained model in the static folder. Please add a new face to continue.')

    else:
        while True:
            # Continuously capture frames from ESP32 camera
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            # Resize and convert the frame to RGB
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
            facesCurFrame = extract_faces(imgS)
        
            for (x, y, w, h) in facesCurFrame:
                face = imgS[y:y+h, x:x+w]
                face = cv2.resize(face, (50, 50))
                face_flatten = face.flatten().reshape(1, -1)
                identified_person = identify_face(face_flatten)[0]
                add_attendance(identified_person)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()
        cv2.imread


def update_student_profile():

    def update_standard(index, df):
        standard = simpledialog.askstring("Input", "Enter standard:").upper()
        if standard:
            df.at[index, 'Standard'] = standard
            df.to_excel(attendance_sheet, index=False)
            messagebox.showinfo("Success", "standard updated")

    def update_photo(first_name, surname):
        cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
        name_drive = 'static/faces/' + first_name + " " + surname
        if not os.path.isdir(name_drive):
            os.makedirs(name_drive)
        else:
            files = glob.glob(name_drive + '/*')
            for f in files:
                os.remove(f)
        
        i, j = 0, 0
        nimgs = 10  # Define the number of images to capture
        messagebox.showinfo("Instructions", "Press 'Spacebar' to capture image or 'ESC' to quit.")
        
        while True:
            # Capture frame from ESP32 camera
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            # Extract faces from the frame
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = first_name + "_" + surname + '_' + str(i) + '.jpg'
                    cv2.imwrite(name_drive + '/' + name, frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:  # Press 'ESC' to break
                break
        cv2.destroyAllWindows()
        train_model()

    def find_student(df, entry_first_name, entry_surname):
        first_name = entry_first_name.get().title()
        surname = entry_surname.get().title()
        
        for index, row in df.iloc[:].iterrows():
            if (row['Name'] == first_name) and (row['Surname'] == surname):
                messagebox.showinfo("Student Found", f"Student {first_name} {surname} is already in the list.")
                
                def on_update_standard():
                    update_standard(index, df)
                    update_window.destroy()
                
                def on_update_photo():
                    update_photo(first_name, surname)
                    update_window.destroy()
                
                update_window = tk.Toplevel()
                update_window.title("Update Options")
                tk.Label(update_window, text="Choose an option to update:").pack(pady=10)
                tk.Button(update_window, text="Update standard", command=on_update_standard).pack(pady=5)
                tk.Button(update_window, text="Update Photo", command=on_update_photo).pack(pady=5)
                tk.Button(update_window, text="Exit", command=update_window.destroy).pack(pady=5)
                break
        else:
            messagebox.showwarning("Not Found", "Person not found")
            return

    df = pd.read_excel(attendance_sheet)
    
    root = tk.Tk()
    root.title("Find Student")
    
    tk.Label(root, text="First Name:").grid(row=0, column=0, padx=10, pady=10)
    entry_first_name = tk.Entry(root)
    entry_first_name.grid(row=0, column=1, padx=10, pady=10)
    
    tk.Label(root, text="Surname:").grid(row=1, column=0, padx=10, pady=10)
    entry_surname = tk.Entry(root)
    entry_surname.grid(row=1, column=1, padx=10, pady=10)
    
    find_button = tk.Button(root, text="Find Student", command=lambda: find_student(df, entry_first_name, entry_surname))
    find_button.grid(row=2, columnspan=2, pady=10)
    
    root.mainloop()

def on_exit():
    messagebox.showinfo("Exit", "Thank you")
    root.destroy()

root = tk.Tk()
root.title("Attendance System")
root.geometry("300x300")



# Create buttons for each option
btn_add_student = tk.Button(root, text="1. Add new Student", command=add_student, width=30, height=2)
btn_add_student.pack(pady=10)

btn_mark_attendance = tk.Button(root, text="2. Mark attendance", command=mark_attendance, width=30, height=2)
btn_mark_attendance.pack(pady=10)

btn_update_student = tk.Button(root, text="3. Find and update Student", command=update_student_profile, width=30, height=2)
btn_update_student.pack(pady=10)

btn_exit = tk.Button(root, text="4. Exit", command=on_exit, width=30, height=2)
btn_exit.pack(pady=10)

# Run the application
root.mainloop()

