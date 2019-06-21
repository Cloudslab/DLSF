package org.cloudbus.cloudsim.power.models;

/**
 * Power consumption of Raspberry Pi 3 Model B
 * @source Kaup, Fabian, Philip Gottschling, and David Hausheer.
 * "PowerPi: Measuring and modeling the power consumption of the Raspberry Pi."
 * In 39th Annual IEEE Conference on Local Computer Networks, pp. 236-243. IEEE, 2014.
 */
public class PowerModelSpecRaspberryPi3B extends PowerModelSpecPower {
    /**
     * The power consumption according to the utilization percentage.
     * @see #getPowerData(int)
     */
    private final double[] power = { 1.57, 1.59, 1.61, 1.62, 1.65, 1.66, 1.69, 1.71, 1.72, 1.74, 1.75 };

    @Override
    protected double getPowerData(int index) {
        return power[index];
    }

}