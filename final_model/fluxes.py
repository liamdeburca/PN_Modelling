def construct_image(ion, wave, overlap_array, rho_array, rho_e_array, T_array, filling_array, distance):

    try:
        label, line_type = ion.possible_lines[wave]
    except:
        print('Invalid wavelength for given ion!')
        return None

    emmisivities = rho_array * rho_e_array * ion.getEmmissivity(label, temperature=T_array, density=rho_e_array, line_type=line_type)

    for _, (emm, shape, filling) in enumerate(zip(emmisivities, overlap_array, filling_array)):
        if _ == 0:
            image = emm * shape * filling * distance.getElementSize()
        else:
            image += emm * shape * filling * distance.getElementSize()

    return image

def construct_all(all_ions, overlap_array, rho_array, rho_e_array, T_array, filling_array, distance):

    images = {}
    for ion, wave, _, _ in all_ions.getPairs():
        name = ion.element + '_' + ion.ionisation + '_' + str(wave)
        images[name] = construct_image(ion, wave, overlap_array, rho_array, rho_e_array, T_array, filling_array, distance)

    return images