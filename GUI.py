from tkinter import *
import agentRevised
import CreationScriptRevised

# black window
root = Tk()


theLabel = Label(root, text="Amadeus", fg="black")
seasonLabel = Label(root, text="Season", fg="black")

season = Entry(root)

button1 = Button(root, text="Creation", fg="black")

button2 = Button(root, text="Recommendation", fg="black")
theLabel.grid(columnspan= 2, row=0 )
seasonLabel.grid(row=1)
season.grid(row=1, column=1)

c = Checkbutton(root, text="With Post-Filtering")
c.grid(columnspan=2)



# mantem a janela aberta
root.mainloop()

