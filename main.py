"""
Simple interactive menu to run the face recognition pipeline end-to-end.

Usage:
    python main.py
"""
import sys

from src import enroll, evaluate, identify


def menu():
    print(
        """
Face Recognition Identification System
1) Enroll faces from data/enrolled_images/<person_name>/
2) Identify a face in an image
3) Run live webcam identification
4) Run evaluation on the LFW dataset
0) Exit
"""
    )


def main():
    while True:
        menu()
        choice = input("Select an option: ").strip()

        if choice == "1":
            enroll.main()
        elif choice == "2":
            path = input("Path to image: ").strip()
            sys.argv = ["identify.py", path, "--save"]
            identify.main()
        elif choice == "3":
            from src import webcam_demo

            webcam_demo.main()
        elif choice == "4":
            evaluate.main()
        elif choice == "0":
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()
