import os
from pathlib import Path

# Partimos de la ruta en la que se encuentre el proyecto.
current_path = Path.cwd()

def show_current_path_files():
    print("Listando archivos del directorio actual:")
    for index, file in enumerate(current_path.iterdir()):
        print(str(index) + ". " + file.name)

def go_parent_dir():
    # Global es para acceder a variables definidas fuera de esta función.
    global current_path
    current_path = current_path.parent
    os.chdir(current_path)

def go_child_dir(dir_name: str):
    global current_path
    new_path_str = os.path.join(current_path, dir_name)
    new_path = Path(new_path_str)
    if new_path.is_dir():
        os.chdir(new_path)
        current_path = new_path
    else:
        print("Error. El archivo seleccionado no es un directorio.")


print("Bienvenido al explorador de archivos.")

while True:
    command = input(str(current_path) + " -> ").casefold().split(" ")
    # TODO match no funciona en python 3.8 (que es como un switch)
    # match command[0]:
    #     case "list":
    #         show_current_path_files()
    #     case "cd":
    #         if command[1] == "..":
    #             go_parent_dir()
    #         else:
    #             if command[1] is not None and command[1] != "":
    #                 go_child_dir(command[1])
    #             else:
    #                 print("El directorio seleccionado debe existir.")
    #     case _:
    #         print("Abandonando programa.")
    #         break