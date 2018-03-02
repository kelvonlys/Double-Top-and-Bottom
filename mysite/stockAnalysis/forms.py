from django import forms

class NameForm(forms.Form):
    stockNum = forms.CharField(label='Stock num', max_length=100)