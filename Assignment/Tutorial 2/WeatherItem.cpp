#include "WeatherItem.h"

WeatherItem::WeatherItem()
{
}


WeatherItem::~WeatherItem()
{
}


string WeatherItem::getWeatherStation() {
	return weatherStation;
}

int WeatherItem::getYearCollected() {
	return yearCollected;
}

int WeatherItem::getMonth() {
	return month;
}

int WeatherItem::getDay() {
	return day;
}

int WeatherItem::getTime() {
	return time;
}

float WeatherItem::getTemp() {
	return temp;
}


void WeatherItem::setWeatherStation(string s) {
	weatherStation = s;
}

void WeatherItem::setYearCollected(int i) {
	yearCollected = i;
}

void WeatherItem::setMonth(int i) {
	month = i;
}

void WeatherItem::setDay(int i) {
	day = i;
}

void WeatherItem::setTime(int i) {
	time = i;
}

void WeatherItem::setTemp(float f) {
	temp = f;
}

