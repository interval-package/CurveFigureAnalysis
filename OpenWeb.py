import webbrowser
import os.path as path


if __name__ == '__main__':
    url = path.join("WedPage", "index.html")
    webbrowser.open(url, new=2)
