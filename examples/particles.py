"""
Beware! you need PyTables >= 2.3 to run this script!
"""

from time import time  # use clock for Win
import numpy
from tables import *

#NEVENTS = 10000
NEVENTS = 20000
MAX_PARTICLES_PER_EVENT = 100

# Particle description
class Particle(IsDescription):
    #event_id    = Int32Col(pos=1, indexed=True) # event id (indexed)
    event_id    = Int32Col(pos=1)               # event id (not indexed)
    particle_id = Int32Col(pos=2)               # particle id in the event
    parent_id   = Int32Col(pos=3)               # the id of the parent particle
                                                # (negative values means no parent)
    momentum    = Float64Col(shape=3, pos=4)    # momentum of the particle
    mass        = Float64Col(pos=5)             # mass of the particle

# # Create a new table for events
t1 = time()
print "Creating a table with %s entries aprox.. Wait please..." % \
      (int(NEVENTS*(MAX_PARTICLES_PER_EVENT/2.)))
fileh = openFile("particles-pro.h5", mode = "w")
group = fileh.createGroup(fileh.root, "events")
table = fileh.createTable(group, 'table', Particle, "A table", Filters(0))
# Choose this line if you want data compression
#table = fileh.createTable(group, 'table', Particle, "A table", Filters(1))

# Fill the table with events
numpy.random.seed(1)  # In order to have reproducible results
particle = table.row
for i in xrange(NEVENTS):
    for j in xrange(numpy.random.randint(0, MAX_PARTICLES_PER_EVENT)):
        particle['event_id']  = i
        particle['particle_id'] = j
        particle['parent_id'] = j - 10     # 10 root particles (max)
        particle['momentum'] = numpy.random.normal(5.0, 2.0, size=3)
        particle['mass'] = numpy.random.normal(500.0, 10.0)
        # This injects the row values.
        particle.append()
table.flush()
print "Added %s entries --- Time: %s sec" % (table.nrows, round((time()-t1), 3))

t1 = time()
print "Creating index..."
table.cols.event_id.createIndex(optlevel=0, verbose=True)
print "Index created --- Time: %s sec" % (round((time()-t1), 3))
# Add the number of events as an attribute
table.attrs.nevents = NEVENTS

fileh.close()

# Open the file en read only mode and start selections
print "Selecting events..."
fileh = openFile("particles-pro.h5", mode = "r")
table = fileh.root.events.table

print "Particles in event 34:",
nrows = 0; t1 = time()
for row in table.where("event_id == 34"):
        nrows += 1
print nrows
print "Done --- Time:", round((time()-t1), 3), "sec"

print "Root particles in event 34:",
nrows = 0; t1 = time()
for row in table.where("event_id == 34"):
    if row['parent_id'] < 0:
        nrows += 1
print nrows
print "Done --- Time:", round((time()-t1), 3), "sec"

print "Sum of masses of root particles in event 34:",
smass = 0.0; t1 = time()
for row in table.where("event_id == 34"):
    if row['parent_id'] < 0:
        smass += row['mass']
print smass
print "Done --- Time:", round((time()-t1), 3), "sec"

print "Sum of masses of daughter particles for particle 3 in event 34:",
smass = 0.0; t1 = time()
for row in table.where("event_id == 34"):
    if row['parent_id'] == 3:
        smass += row['mass']
print smass
print "Done --- Time:", round((time()-t1), 3), "sec"

print "Sum of module of momentum for particle 3 in event 34:",
smomentum = 0.0; t1 = time()
#for row in table.where("(event_id == 34) & ((parent_id) == 3)"):
for row in table.where("event_id == 34"):
    if row['parent_id'] == 3:
        smomentum += numpy.sqrt(numpy.add.reduce(row['momentum']**2))
print smomentum
print "Done --- Time:", round((time()-t1), 3), "sec"

# This is the same than above, but using generator expressions
# Python 2.4 needed here!
print "Sum of module of momentum for particle 3 in event 34 (2):",
t1 = time()
print sum(numpy.sqrt(numpy.add.reduce(row['momentum']**2))
          for row in table.where("event_id == 34")
          if row['parent_id'] == 3)
print "Done --- Time:", round((time()-t1), 3), "sec"


fileh.close()
