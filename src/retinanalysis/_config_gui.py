"""Standalone config GUI script to get around stupid issues with
TKinter on Mac, which is a nightmare and its maintainers should be
ashamed of themselves.
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
    'meta',
    'tags',
    'query',
    'user'
)

KEY_DICT = {
    'analysis' : 'Analysis directory',
    'data' : 'Sorted data directory',
    'raw' : 'Raw data directory',
    'h5' : 'H5 file directory',
    'meta' : 'Metadata directory',
    'tags' : 'Tags directory',
    'query' : 'Query directory',
    'user' : 'Username',
}


def main(result_file):
    root = tk.Tk()
    root.title('Retinanalysis config setup')


    # Create the grid for buttons and such
    frame = ttk.Frame(root, padding=20)
    frame.grid()

    name_var = tk.StringVar()
    ttk.Label(frame, text=f'Profile name: ').grid(column=0, row=0, sticky='w')
    ttk.Entry(frame, textvariable=name_var).grid(column=1, row=0, columnspan=2)

    # Fill all the entries
    entries=dict()
    for idx, key in enumerate(REQUIRED_KEYS):
        ttk.Label(frame, text=f'{KEY_DICT[key]} :').grid(column=0, row=idx+1, sticky = 'w')

        var = tk.StringVar()
        entries[key] = var

        if key != 'user':
            ttk.Entry(frame, textvariable=entries[key]).grid(column=1,row=idx+1)
            def on_browse(v=var):
                path = filedialog.askdirectory()
                if path:
                    v.set(path)
                
            ttk.Button(frame, text='Browse', command=on_browse).grid(column=2, row=idx+1)
        else:
            ttk.Entry(frame, textvariable=entries[key]).grid(column=1, row=idx+1, columnspan=2)


    def create_callback():
        name = name_var.get()
        if not name:
            messagebox.showwarning(
                title='Empty Fields',
                message='Profile name field is empty, please fill out all fields.',
            )
            return

        profile_paths = {key: var.get() for key, var in entries.items()}
        for key, var in profile_paths.items():
            if not var:
                messagebox.showwarning(
                    title='Empty Fields',
                    message=f'No {KEY_DICT[key]} value given, please fill out all fields',
                )
                return

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
