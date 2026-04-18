# Packaging

## 1) Build EXE (PyInstaller)

Use the helper script:

```bat
build_exe.bat
```

Or run directly:

```bat
.venv\Scripts\python.exe -m PyInstaller --noconfirm yolov26_l.spec
```

Output:

- dist\yolov26_l\yolov26_l.exe

## 2) Build Installer (Inno Setup)

Prerequisite:

- Install Inno Setup 6 (ISCC.exe)
- Default path used by script:
  - C:\Program Files (x86)\Inno Setup 6\ISCC.exe

Build command:

```bat
build_installer.bat
```

Or run directly:

```bat
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\yolov26_l.iss
```

Output:

- dist_installer\YOLOV26_L_Setup.exe

## Notes

- The installer script packages everything under dist\yolov26_l\.
- If you move ISCC.exe, edit build_installer.bat accordingly.
- If you need a custom icon, set SetupIconFile and executable icon in your build process.
