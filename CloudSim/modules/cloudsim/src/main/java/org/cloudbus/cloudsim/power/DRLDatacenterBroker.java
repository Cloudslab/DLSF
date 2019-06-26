package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.lists.VmList;
import org.cloudbus.cloudsim.plus.DatacenterBrokerEX;
import org.cloudbus.cloudsim.plus.util.CustomLog;

import java.util.List;
import java.util.logging.Level;

public class DRLDatacenterBroker extends DatacenterBrokerEX {

    public DRLDatacenterBroker(String name) throws Exception {
        super(name, -1);
    }

    /**
     * Submits the list of vms after a given delay
     *
     * @param vms list of vms
     * @param delay delay of creation
     */
    @Override
    public void createVmsAfter(List<? extends Vm> vms, double delay) {
        if(getVmList().size() > 90){
            return;
        }
        if (started) {
            send(getId(), delay, BROKER_SUBMIT_VMS_NOW, vms);
        } else {
            presetEvent(getId(), BROKER_SUBMIT_VMS_NOW, vms, delay);
        }
    }

    /**
     * Submits the cloudlets after a specified time period. Used mostly for
     * testing purposes.
     *
     * @param cloudlets
     *            - the cloudlets to submit.
     * @param delay
     *            - the delay.
     */
    @Override
    public void submitCloudletList(List<? extends Cloudlet> cloudlets, double delay) {
        if(getVmList().size() > 90){
            return;
        }
        if (started) {
            send(getId(), delay, BROKER_CLOUDLETS_NOW, cloudlets);
        } else {
            presetEvent(getId(), BROKER_CLOUDLETS_NOW, cloudlets, delay);
        }
    }

}
