class Readout:
    def __init__(self, config):
        """
        Initialize a Readout instance with specific properties from a
        dictionary.

        Args:
            config (dict): A dictionary containing key-value pairs for the Readout properties.

        Raises:
            ValueError: If a key in the dictionary is not an allowed attribute.
        """
        self.allowed_attributes = {
            "RO_LO",
            "RO_LO_34",
            "RO_LO_45",
            "RO_LO_pwr",
            "ro_dur",
        }
        # Validate and set attributes from the dictionary
        for key, value in config.items():
            if key not in self.allowed_attributes:
                raise ValueError(
                    f"Invalid attribute '{key}' for Readout. Allowed attributes are: {self.allowed_attributes}"
                )
            setattr(self, key, value)
        # Set default values (None) for missing attributes
        for attr in self.allowed_attributes:
            if not hasattr(self, attr):
                setattr(self, attr, None)

    def __repr__(self):
        attributes = ", ".join(
            f"{attr}={getattr(self, attr)}" for attr in self.allowed_attributes
        )
        return f"Readout({attributes})"


class Qubit(Readout):
    def __init__(self, qubit_config, readout_config):
        """
        Initialize a Qubit instance with both Qubit- and Readout-related
        configuration. The Qubit class inherits from Readout so that the
        readout parameters are automatically initialized via a super() call.

        If the Qubit configuration does not include 'ROIF', it is automatically computed as:
            ROIF = ro_freq - RO_LO

        Args:
            qubit_config (dict): A dictionary with Qubit-specific properties.
            readout_config (dict): A dictionary with Readout-specific properties.

        Raises:
            ValueError: If a key in the configuration is invalid or if necessary values are missing.
        """
        # Initialize the readout part using the parent class constructor
        super().__init__(readout_config)

        # Define Qubit-specific allowed attributes (excluding ROIF, which will be computed)
        self.allowed_qubit_attributes = {
            "qubit_id",
            "ro_freq",
            "ge_time",
            "ef_time",
            "ef_half_time",
            "ge_ssm",
            "ef_ssm",
            "ge_amp",
            "ef_amp",
            "ef_half_amp",
            "IQ_angle",
            "ro_amp",
            "qubit_thr",
            "RO_IF",
            "mixer_offset_ge",
            "mixer_offset_ef",

        }

        # Validate and set Qubit-specific attributes
        for key, value in qubit_config.items():
            if key not in self.allowed_qubit_attributes:
                raise ValueError(
                    f"Invalid attribute '{key}' for Qubit. Allowed attributes are: {self.allowed_qubit_attributes}"
                )
            setattr(self, key, value)
        # Set defaults (None) for any missing Qubit-specific attributes
        for attr in self.allowed_qubit_attributes:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        # Automatically compute ROIF if not provided.
        # (Since ROIF is not a Qubit-specific allowed attribute, it will be computed here.)
        if self.ro_freq is None or self.RO_LO is None:
            raise ValueError(
                "Cannot compute ROIF because either 'ro_freq' or 'RO_LO' is missing."
            )
        self.ROIF = self.ro_freq - self.RO_LO

    def __repr__(self):
        # Combine both Qubit and Readout attributes plus the computed ROIF
        combined_attrs = self.allowed_qubit_attributes.union(
            self.allowed_attributes
        ).union({"ROIF"})
        attributes = ", ".join(
            f"{attr}={getattr(self, attr)}" for attr in combined_attrs
        )
        return f"Qubit({attributes})"
