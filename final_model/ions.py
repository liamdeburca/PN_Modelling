class Ion:
    def __init__(self, element, ionisation, possible_lines = {}):
        from pyneb import Atom, RecAtom

        self.element = element
        self.ionisation = ionisation
        self.possible_lines = possible_lines

        try: 
            self.coll = Atom(element, ionisation)
        except:
            self.coll = None

        try:
            self.rec = RecAtom(element, ionisation)
        except:
            self.rec = None

    def getEmissivity(self, label: str, temperature: float = 1e4, density: float = 1e3, line_type: str = 'collisional'):

        if line_type == 'collisional':
            emission_object = self.coll
        elif line_type == 'recombination':
            emission_object = self.rec
        else:
            print('Please choose a valid line type! Either collisional or recombination...')
            return -1

        if len(label.split('_')) > 1:
            lev_i, lev_j = label.split('_')
            return emission_object.getEmissivity(temperature, density, lev_i=int(lev_i), lev_j=int(lev_j), product=False)
        
        return emission_object.getEmissivity(temperature, density, label=label, product=False)
    
class AllIons:
    """
    Object containing all Ion objects, for easier implementation. 
    """
    def __init__(self):
        # Hydrogen
        H1 = Ion('H', 1, possible_lines={4861: ['4_2', 'recombination'], 6563: ['3_2', 'recombination']})
        self.H1 = H1

        # Helium
        He1 = Ion('He', 1, possible_lines={7065: ['7065.0', 'recombination']})
        He2 = Ion('He', 2, possible_lines={5411: ['7_4', 'recombination']})
        self.He1, self.He2 = He1, He2

        # Nitrogen
        N1 = Ion('N', 1, possible_lines={5199: ['3_1', 'collisional']})
        N2 = Ion('N', 2, possible_lines={5755: ['5_4', 'collisional'], 6548: ['4_2', 'collisional'], 6584: ['4_3', 'collisional']})
        self.N1, self.N2 = N1, N2

        # Oxygen
        O1 = Ion('O', 1, possible_lines={6300: ['4_1', 'collisional']})
        O2 = Ion('O', 2, possible_lines={7320: ['4_2', 'collisional']})
        O3 = Ion('O', 3, possible_lines={5949: ['4_2', 'collisional'], 5007: ['5007.', 'recombination']})
        self.O1, self.O2, self.O3 = O1, O2, O3

        # Sulphur
        S2 = Ion('S', 2, possible_lines={6716: ['3_1', 'collisional'], 6731: ['2_1', 'collisional']})
        S3 = Ion('S', 3, possible_lines={6312: ['5_4', 'collisional'], 9069: ['4_2', 'collisional']})
        self.S2, self.S3 = S2, S3

        # Chlorine
        Cl3 = Ion('Cl', 3, possible_lines={5518: ['3_1', 'collisional'], 5538: ['2_1', 'collisional']})
        Cl4 = Ion('Cl', 4, possible_lines={8045: ['4_3', 'collisional']})
        self.Cl3, self.Cl4 = Cl3, Cl4

        # Argon
        Ar3 = Ion('Ar', 3, possible_lines={7136: ['4_1', 'collisional'], 7751: ['4_2', 'collisional']})
        self.Ar3 = Ar3

    def intoList(self):
        return [self.H1, self.He1, self.He2, self.N1, self.N2, self.O1, self.O2, self.O3, self.S2, self.S3, self.Cl3, self.Cl4, self.Ar3]
    
    def getPairs(self):
        ion_list = self.intoList()

        ions, waves, labels, line_types = [], [], [], []
        for ion in ion_list:
            ws = ion.possible_lines.keys()
            values = ion.possible_lines.values()
            for w, value in zip(ws, values):
                ions.append(ion)
                waves.append(w)
                labels.append(value[0])
                line_types.append(value[1])

        return zip(ions, waves, labels, line_types)