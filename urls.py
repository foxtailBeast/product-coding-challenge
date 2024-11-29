from django.urls import include, path


urlpatterns = [
    path('forms/', include('apps.form.urls')),
]
