"""This program shows the different protections that PyTables offer to
the user in order to insure a correct data injection in tables.

Example to be used in the second tutorial in the User's Guide.

"""

import warnings
from tables import *

class Particle(IsColDescr):
    name      = Col('CharType', 16)  # 16-character String
    lati      = Col("Int32", 1)      # integer
    longi     = Col("Int32", 1)      # integer
    pressure  = Col("Float32", 1)    # float  (single-precision)
    temperature = Col("Float64", 1)    # double (double-precision)

class Event(IsColDescr):
    name      = Col('CharType', 16)    # 16-character String
    TDCcount  = Col("UInt8", 1)        # unsigned byte
    ADCcount  = Col("UInt16", 1)       # Unsigned short integer
    xcoord    = Col("Float32", 1)      # integer
    ycoord    = Col("Float32", 1)      # integer

# Open a file in "w"rite mode
fileh = openFile("tutorial2.h5", mode = "w")
# Get the HDF5 root group
root = fileh.root

warnings.resetwarnings()

# Create the groups:
for groupname in ("Particles", "Events"):
    group = fileh.createGroup(root, groupname)

# Now, create and fill the tables in Particles group
gparticles = root.Particles
# Create 3 new tables
for tablename in ("TParticle1", "TParticle2", "TParticle3"):
    # Create a table
    table = fileh.createTable("/Particles", tablename, Particle(),
                           "Particles: "+tablename)
    # Get the record object associated with the table:
    particle = table.row
    # Fill the table with 10 particles
    for i in xrange(257):
        # First, assign the values to the Particle record
        particle['name'] = 'Particle: %6d' % (i)
        particle['lati'] = i 
        particle['longi'] = 10 - i
        particle['pressure'] = float(i*i)
        particle['temperature'] = float(i**2)
        # This injects the Record values
        particle.append()      

    # Flush the table buffers
    table.flush()

# Now, go for Events:
for tablename in ("TEvent1", "TEvent2", "TEvent3"):
    # Create a table in Events group
    table = fileh.createTable(root.Events, tablename, Event(),
                           "Events: "+tablename)
    # Get the record object associated with the table:
    event = table.row
    # Fill the table with 257 events
    for i in xrange(257):
        # First, assign the values to the Event record
        event['name']  = 'Event: %6d' % (i)
        ########### Errors start here. Play with them!
        # Range checks no longer works on 0.3
        event['TDCcount'] = i            # Wrong range.
        #event['ADCcount'] = i * 2        # Correct type
        #event['xcoor'] = float(i**2)     # Wrong spelling. This works on 0.3
        #event['TDCcount'] = i % (1<<8)   # Correct range
        #event['ADCcount'] = str(i)      # Wrong range
        event['xcoord'] = float(i**2)   # Correct spelling
        ########### End of errors
        event['ycoord'] = float(i)**4
        # This injects the Record values
        event.append()

    # Flush the buffers
    table.flush()

# Read the records from table "/Events/TEvent3" and select some
table = root.Events.TEvent3
e = [ p['TDCcount'] for p in table.iterrows()
      if p['ADCcount'] < 20 and 4 <= p['TDCcount'] < 15 ]
print "Last record ==>", p
print "Selected values ==>", e
print "Total selected records ==> ", len(e)

# Finally, close the file (this also will flush all the remaining buffers!)
fileh.close()
