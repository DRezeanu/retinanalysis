"""Standalone config GUI script to get around stupid issues with
TKinter on Mac, which is a nightmare and its maintainers should be
ashamed of themselves.

This script is used in two ways: to create a config profile or to edit
an existing one. If the input file is empty, we default to creating a 
profile. If it's not, we default to editing. 
"""
import tkinter as tk
from tkinter import (
    ttk,
    filedialog,
    messagebox,
)
import json
import sys

REQUIRED_KEYS = (
    'analysis',
    'data',
    'raw', 
    'h5',
    'vision',
    'meta',
    'tags',
    'user'
)

KEY_DICT = {
    'analysis' : 'Analysis directory',
    'data' : 'Sorted data directory',
    'raw' : 'Raw data directory',
    'h5' : 'H5 file directory',
    'vision' : 'Path to Vision.jar',
    'meta' : 'Metadata directory',
    'tags' : 'Tags directory',
    'user' : 'Username',
}


def main(result_file):

    with open(result_file, 'r') as f:
        input_dict = json.load(f)

    root = tk.Tk()
    root.title('Retinanalysis config setup')

    # Put the window above the current active window
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()
    root.after(100, lambda: root.attributes('-topmost', False))

    # Create the grid for buttons and such
    frame = ttk.Frame(root, padding=20)
    frame.grid()

    # Pull profile name
    if not input_dict:
        name_var = tk.StringVar()
        ttk.Label(frame, text=f'Profile name: ').grid(column=0, row=0, sticky='w')
        ttk.Entry(frame, textvariable=name_var).grid(column=1, row=0, columnspan=2)
    else:
        name_var = tk.StringVar(value=input_dict['name'])
        ttk.Label(frame, text=f'Profile name: ').grid(column=0, row=0, sticky='w')
        ttk.Entry(frame, textvariable=name_var).grid(column=1, row=0, columnspan=2)


    # Fill all the entries
    entries=dict()
    for idx, key in enumerate(REQUIRED_KEYS):
        ttk.Label(frame, text=f'{KEY_DICT[key]} :').grid(column=0, row=idx+1, sticky = 'w')

        if not input_dict:
            var = tk.StringVar()
            entries[key] = var
        else:
            var = tk.StringVar(value=input_dict[key])
            entries[key] = var

        if key == 'user':
            ttk.Entry(frame, textvariable=entries[key]).grid(column=1, row=idx+1, columnspan=2)

        elif key == 'vision':
            ttk.Entry(frame, textvariable=entries[key]).grid(column=1, row=idx+1)
            def on_browse_file(v=var):
                path = filedialog.askopenfilename()
                if path:
                    v.set(path)
            ttk.Button(frame, text='Browse', command=on_browse_file).grid(column=2, row=idx+1)

        else:
            ttk.Entry(frame, textvariable=entries[key]).grid(column=1,row=idx+1)
            def on_browse_dir(v=var):
                path = filedialog.askdirectory()
                if path:
                    v.set(path)
            ttk.Button(frame, text='Browse', command=on_browse_dir).grid(column=2, row=idx+1)

    # Write an empty string as path if no path given for a particular
    # entry. This allows users to leave certain paths blank if they wish.
    # The only required fields are the profile name and the username
    def create_callback():
        name = name_var.get()
        if not name:
            messagebox.showwarning(
                title='Required Fields',
                message='Profile name and Username are required.',
            )
            return

        profile_paths = {key: var.get() for key, var in entries.items()}
        for key, var in profile_paths.items():
            if key == 'user':
                if not var:
                    messagebox.showwarning(
                        title='Required Fields',
                        message='Profile name and Username are required.',
                    )
                    return
            else:
                if not var:
                    profile_paths[key] = ""

        with open(result_file, 'w') as f:
            json.dump({'name':name, 'paths':profile_paths}, f)

        root.destroy()

    ttk.Button(frame, text='Create', command=create_callback).grid(column=0, row=len(REQUIRED_KEYS)+3, columnspan=3)

    # Force the window to make a geometry calculation
    root.update_idletasks()

    # Center the window
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
    root.geometry(f"+{x}+{y}")

    # Create the window and run the main loop

    root.mainloop()



if __name__=='__main__':
    main(sys.argv[1])
