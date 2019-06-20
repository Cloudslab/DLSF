package org.cloudbus.cloudsim.cost.models;

/**
 * The CostModel interface needs to be implemented in order to provide a model of cost
 * of hosts, depending on utilization of a critical system component, such as CPU, RAM and network.
 *
 * @author Shreshth Tuli
 * @since CloudSim Toolkit 5.0
 */

public class CostModel {

    /** Cost per Bw **/
    private double costPerBw = 0;

    /** Cost per CPU time in seconds **/
    private double costPerCPUtime = 0;

    /** Cost per RAM usage **/
    private double costPerRam = 0;

    /**
     * Gets cost to use each MegaBit of bandwidth
     * @return the cost to use bandwidth
     */
    protected double getCostPerBw(){
        return costPerBw;
    };

    /**
     * Gets cost to use CPU
     * @return the cost to use cpu
     */
    protected double getCostPerCPUtime(){
        return costPerCPUtime;
    };

    /**
     * Gets cost to use RAM
     * @return the cost to ram
     */
    protected double getCostPerRam(){
        return costPerRam;
    };
}
