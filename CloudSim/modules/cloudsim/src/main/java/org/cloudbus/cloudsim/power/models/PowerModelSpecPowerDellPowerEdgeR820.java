package org.cloudbus.cloudsim.power.models;

/**
 * The power model of an Dell Power Edge R820 (Intel Xeon E5-4650L 2.60 GHz)
 * *The data is accessed from https://www.spec.org/power_ssj2008/results/res2014q4/power_ssj2008-20141023-00677.html
 * @author Shreshth Tuli
 * @since CloudSim Toolkit 5.0
 *
 * */

public class PowerModelSpecPowerDellPowerEdgeR820  extends PowerModelSpecPower {

    /** The power.
     */
    private final double[] power = { 110, 149, 167, 188, 218, 237, 268, 307, 358, 414, 446};

    /*
     * (non-Javadoc)
     * @see org.cloudbus.cloudsim.power.models.PowerModelSpecPower#getPowerData(int)
     */
    @Override
    protected double getPowerData(int index) {
        return power[index];
    }
}