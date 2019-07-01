package org.cloudbus.cloudsim.power.models;

/**
 * The power model of an DEPO Race X340H (Intel Core i5-4570 4 core 3.20 GHz)
 * *The data is accessed from https://www.spec.org/power_ssj2008/results/res2014q4/power_ssj2008-20141023-00677.html
 * @author Shreshth Tuli
 * @since CloudSim Toolkit 5.0
 *
 * */

public class PowerModelSpecPowerDEPORaceX340H  extends PowerModelSpecPower {

    /** The power.
     */
    private final double[] power = { 83.2, 88.2, 94.3, 101, 107, 112, 117, 120, 124, 128, 131};

    /*
     * (non-Javadoc)
     * @see org.cloudbus.cloudsim.power.models.PowerModelSpecPower#getPowerData(int)
     */
    @Override
    protected double getPowerData(int index) {
        return power[index];
    }
}