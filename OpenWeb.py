import webbrowser
import os.path as path


def Open_Web():
    url = path.join("WedPage", "index.html")
    webbrowser.open(url, new=2)
    pass


if __name__ == '__main__':
    Open_Web()
    pass
