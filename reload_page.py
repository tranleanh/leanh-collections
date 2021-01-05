from subprocess import Popen
import time

urls = ["https://github.com/tranleanh", "https://github.com/tranleanh"]

i=1

while i>0:

	for url in urls:
	    Popen(['start', 'chrome' , url], shell=True)

	time.sleep(5)

	Popen('taskkill /F /IM chrome.exe', shell=True)