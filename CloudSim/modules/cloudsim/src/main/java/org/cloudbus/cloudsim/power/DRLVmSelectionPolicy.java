package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.Vm;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class DRLVmSelectionPolicy extends PowerVmSelectionPolicy{

    /**
     * Gets the list of migratable VMs from a given host.
     *
     * @return the list of migratable VMs
     */
    protected List<PowerVm> getAllMigratableVms(List<PowerHost> hosts) {
        List<PowerVm> migratableVms = new ArrayList<PowerVm>();
        for(PowerHost host : hosts){
            for (PowerVm vm : host.<PowerVm> getVmList()) {
                if (!vm.isInMigration()) {
                    migratableVms.add(vm);
                }
            }
        }
        return migratableVms;
    }

    /**
     * Gets a VM to migrate from a given host.
     *
     * @param host the host
     * @return the vm to migrate
     */
    @Override
    public Vm getVmToMigrate(PowerHost host){
        List<Vm> output = new LinkedList<Vm>();
        List<Vm> vmsToMigrate = new LinkedList<Vm>();
        // Parse output from DL model
        for (Vm vm : output) {
            if(!vm.isInMigration()){
                vmsToMigrate.add(vm);
            }
        }
        return ((LinkedList<Vm>) vmsToMigrate).getFirst();
    }

    /**
     * Gets the VMs to migrate from hosts.
     *
     * @return the VMs to migrate from hosts
     */
    protected List<? extends Vm>
    getAllVmaToMigrate(List<PowerHost> hosts) {
        List<Vm> output = new LinkedList<Vm>();
        List<Vm> vmsToMigrate = new LinkedList<Vm>();
        List<PowerVm> migratableVms = getAllMigratableVms(hosts);
        // @todo : Parse output from DL model
        for (Vm vm : output) {
            if(migratableVms.contains(vm)){
                vmsToMigrate.add(vm);
            }
        }
        return vmsToMigrate;
    }
}
