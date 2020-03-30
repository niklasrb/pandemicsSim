from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sim import *



fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.2, bottom=0.25)


disease = Disease(0.5, 4., 0.5, 14., 0.2, 0.3)
community = Community(250000., 3., 0.03)

y0 = initialConditions(community)
tfin = 100.
t, ys = simulate(disease, community, y0, tfin)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
ax.set_xlabel("Time / days")
ax.set_ylabel("Population")
plotCommunity(ax, community, t, ys, disease)

# Add two sliders for tweaking the parameters_
sliderSocDisAx  = fig.add_axes([0.2, 0.15, 0.65, 0.03])
sliderSocDis  = Slider(sliderSocDisAx, 'Social Distancing', 0., 1., valinit=community.socialDistancing)

# Draw another slider
sliderCareAx = fig.add_axes([0.2, 0.1, 0.65, 0.03])
sliderCare = Slider(sliderCareAx, 'Care Capacity', 0., 0.1, valinit=community.maxCareCapacity)

sliderTfinAx = fig.add_axes([0.2, 0.05, 0.65, 0.03])
sliderTfin = Slider(sliderTfinAx, 't_final', 10, 1000, valinit=tfin)

radioQuarantineAx = fig.add_axes([0.025, 0.5, 0.15, 0.1])
radioQuarantine = RadioButtons(radioQuarantineAx, ('0', '1', '2', '3'), active=0)

sliderQuarEffAx = fig.add_axes([0.025, 0.3, 0.15, 0.03])
sliderQuarEff = Slider(sliderQuarEffAx, 'quarantine Effectivness', 0., 1., valinit=community.quarantineEffectiveness)

# Define an action for modifying the line when any slider's value changes
def on_changed(val):
    community.maxCareCapacity = sliderCare.val
    community.socialDistancing = sliderSocDis.val
    community.quarantineMeasures = int(radioQuarantine.value_selected)
    community.quarantineEffectiveness = sliderQuarEff.val
    tfin = sliderTfin.val
    t, ys = simulate(disease, community, y0, tfin)
    ax.clear()
    plotCommunity(ax, community, t, ys, disease)
    fig.canvas.draw_idle()

sliderSocDis.on_changed(on_changed)
sliderCare.on_changed(on_changed)
sliderTfin.on_changed(on_changed)
sliderQuarEff.on_changed(on_changed)

radioQuarantine.on_clicked(on_changed)
# Add a button for resetting the parameters
# reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
# reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
# def reset_button_on_clicked(mouse_event):
#     freq_slider.reset()
#     amp_slider.reset()
# reset_button.on_clicked(reset_button_on_clicked)

# Add a set of radio buttons for changing color

plt.show()
