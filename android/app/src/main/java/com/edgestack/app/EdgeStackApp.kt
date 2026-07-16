package com.edgestack.app

import android.app.Application
import com.edgestack.app.di.AppContainer
import com.edgestack.app.work.AlertNotifier
import com.edgestack.app.work.WorkScheduler

class EdgeStackApp : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        AlertNotifier.createChannels(this)
        WorkScheduler.scheduleNext(this, container.calendarRepo.calendar)
    }
}
